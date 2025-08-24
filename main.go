package main

import (
	"bufio"
	"bytes"
	"cmp"
	_ "embed"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	mapset "github.com/deckarep/golang-set/v2"
	"github.com/ethereum/go-ethereum/cmd/utils"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/prque"
	"github.com/ethereum/go-ethereum/core/rawdb"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/rlp"
	"github.com/gabriel-vasile/mimetype"
	"github.com/klauspost/compress/gzip"
	"github.com/klauspost/compress/snappy"
	"github.com/klauspost/compress/zstd"
	"github.com/klauspost/reedsolomon"
	"github.com/ulikunitz/xz"
	"github.com/urfave/cli/v2"
	"golang.org/x/sync/errgroup"
)

//go:embed network.py
var networkScriptContent string

//go:embed graph.py
var graphScriptContent string

var app = cli.NewApp()

var (
	preprocessCommand = &cli.Command{
		Action: preprocess,
		Name:   "preprocess",
		Flags: []cli.Flag{
			utils.DataDirFlag,
		},
	}
	simulateCorrectionFactorCommand = &cli.Command{
		Action: simulateCorrectionFactor,
		Name:   "simulate-correction-factor",
	}
	simulateScalabilityCommand = &cli.Command{
		Action: simulateScalability,
		Name:   "simulate-scalability",
	}
	analyzeScalabilityCommand = &cli.Command{
		Action: analyzeScalability,
		Name:   "analyze-scalability",
	}
	simulateBlockSizeCommand = &cli.Command{
		Action: simulateBlockSize,
		Name:   "simulate-block-size",
	}
	analyzeBlockSizeCommand = &cli.Command{
		Action: analyzeBlockSize,
		Name:   "analyze-block-size",
	}
	simulateBandwidthCommand = &cli.Command{
		Action: simulateBandwidth,
		Name:   "simulate-bandwidth",
	}
	analyzeBandwidthCommand = &cli.Command{
		Action: analyzeBandwidth,
		Name:   "analyze-bandwidth",
	}
	simulateSimilarityCommand = &cli.Command{
		Action: simulateSimilarity,
		Name:   "simulate-similarity",
	}
	analyzeSimilarityCommand = &cli.Command{
		Action: analyzeSimilarity,
		Name:   "analyze-similarity",
	}
)

func init() {
	app.Commands = []*cli.Command{
		preprocessCommand,
		simulateCorrectionFactorCommand,
		simulateScalabilityCommand,
		analyzeScalabilityCommand,
		simulateBlockSizeCommand,
		analyzeBlockSizeCommand,
		simulateBandwidthCommand,
		analyzeBandwidthCommand,
		simulateSimilarityCommand,
		analyzeSimilarityCommand,
	}
}

type Reader struct {
	file   *os.File
	reader io.Reader
}

func (r *Reader) Read(p []byte) (n int, err error) {
	return r.reader.Read(p)
}

func (r *Reader) Close() error {
	return r.file.Close()
}

func newReader(path string) (*Reader, error) {
	mime, err := mimetype.DetectFile(path)
	if err != nil {
		return nil, err
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	var reader io.Reader
	switch {
	case mime.Is("application/gzip"):
		reader, err = gzip.NewReader(file)
		if err != nil {
			return nil, err
		}
	case mime.Is("application/x-xz"):
		reader, err = xz.NewReader(file)
		if err != nil {
			return nil, err
		}
	case mime.Is("application/zstd"):
		reader, err = zstd.NewReader(file)
		if err != nil {
			return nil, err
		}
	case mime.Is("application/octet-stream"):
		fallthrough
	case mime.Is("text/plain"):
		reader = bufio.NewReader(file)
	default:
		return nil, errors.New("unsupported file type: " + mime.String())
	}
	return &Reader{
		file:   file,
		reader: reader,
	}, nil
}

func readTxPool(path string) (map[uint64]mapset.Set[common.Hash], error) {
	txpools := make(map[uint64]mapset.Set[common.Hash])
	reader, err := newReader(path)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 512*1024), 512*1024)
	for scanner.Scan() {
		line := scanner.Text()
		words := strings.Split(strings.Trim(line, "\n"), ",")
		height, err := strconv.ParseUint(words[0], 10, 64)
		if err != nil {
			return nil, err
		}
		txpool := mapset.NewSet[common.Hash]()
		for _, word := range words[1:] {
			hash := common.HexToHash(word)
			txpool.Add(hash)
		}
		txpools[height] = txpool
	}
	return txpools, nil
}

func readBlocks(path string) (map[uint64]*types.Block, error) {
	reader, err := newReader(path)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	var blockList []*types.Block
	if err = rlp.Decode(reader, &blockList); err != nil {
		return nil, err
	}
	blocks := make(map[uint64]*types.Block)
	for _, block := range blockList {
		blocks[block.NumberU64()] = block
	}
	return blocks, nil
}

type BlockInfo struct {
	Heights      [2]uint64
	OriginalSize uint64
	NativeSize   uint64
	AliasSize    uint64
	BCBSize      uint64
	BCBHitRate   float64
	ECCBSize     map[string]uint64
	ECCBHitRate  map[string]float64
	HitTxRate    float64
}

type CompactBlock struct {
	header      *types.Header
	uncles      []*types.Header
	txHashes    []common.Hash
	txSizes     []uint64
	chunkSize   uint64
	parity      []byte
	withdrawals types.Withdrawals
}

type extCompactBlock struct {
	Header      *types.Header
	Uncles      []*types.Header
	TxHashes    []common.Hash
	TxSizes     []uint64
	ChunkSize   uint64
	Parities    []byte
	Withdrawals []*types.Withdrawal `rlp:"optional"`
}

func (cb *CompactBlock) EncodeRLP(w io.Writer) error {
	return rlp.Encode(w, &extCompactBlock{
		Header:      cb.header,
		TxHashes:    cb.txHashes,
		Uncles:      cb.uncles,
		Withdrawals: cb.withdrawals,
		TxSizes:     cb.txSizes,
		ChunkSize:   cb.chunkSize,
		Parities:    cb.parity,
	})
}

const (
	START uint64 = 19130000
	END   uint64 = 19140000
)

func preprocess(ctx *cli.Context) error {
	var blocks map[uint64]*types.Block
	if ctx.Args().Len() == 3 {
		var err error
		blocks, err = readBlocks(ctx.Args().Get(0))
		if err != nil {
			return err
		}
	} else {
		dataDir := ctx.String(utils.DataDirFlag.Name)
		db, err := rawdb.Open(rawdb.OpenOptions{
			Directory:         path.Join(dataDir, "geth", "chaindata"),
			AncientsDirectory: path.Join(dataDir, "geth", "chaindata", "ancient"),
			ReadOnly:          true,
		})
		if err != nil {
			return err
		}
		for height := START; height < END; height++ {
			hash := rawdb.ReadCanonicalHash(db, height)
			block := rawdb.ReadBlock(db, hash, height)
			blocks[height] = block
		}
		if err = db.Close(); err != nil {
			return err
		}
	}

	var txpoolsList [2]map[uint64]mapset.Set[common.Hash]
	for i := 0; i < 2; i++ {
		var err error
		if ctx.Args().Len() == 3 {
			txpoolsList[i], err = readTxPool(ctx.Args().Get(i + 1))
			if err != nil {
				return err
			}
		} else {
			txpoolsList[i], err = readTxPool(ctx.Args().Get(i))
			if err != nil {
				return err
			}
		}
	}

	rates := []float64{
		1.00, 1.05, 1.10, 1.15, 1.16, 1.17, 1.18, 1.19, 1.20, 1.25, 1.30,
		1.35, 1.40,
	}

	for i, interval := range [][2]uint64{
		{START, START + (END-START)/4},
		{START + (END-START)/4, START + (END-START)/2},
		{START + (END-START)/2, START + (END-START)/4*3},
		{START + (END-START)/4*3, END},
	} {
		var nativeSize atomic.Uint64
		var aliasSize atomic.Uint64
		var bcbSize atomic.Uint64
		var bcbHitNum atomic.Uint64
		var totalSize atomic.Uint64
		var hitSize atomic.Uint64
		eccbSize := make([]atomic.Uint64, len(rates))
		eccbHitNum := make([]atomic.Uint64, len(rates))

		var wg errgroup.Group
		var cur atomic.Uint64
		cur.Store(interval[0])
		for t := 0; t < runtime.NumCPU(); t++ {
			wg.Go(func() error {
				for height := cur.Add(1) - 1; height < interval[1]; height = cur.Add(1) - 1 {
					block := blocks[height]
					blockTxs := mapset.NewThreadUnsafeSet[common.Hash]()
					for _, tx := range block.Transactions() {
						blockTxs.Add(tx.Hash())
					}

					txSizeMap := make(map[common.Hash]int)
					for _, tx := range block.Transactions() {
						totalSize.Add(tx.Size() * 2)
						txBytes, err := rlp.EncodeToBytes(tx)
						if err != nil {
							return err
						}
						txBytes = snappy.Encode(txBytes, txBytes)
						txSizeMap[tx.Hash()] = len(txBytes)
					}

					blockBytes, err := rlp.EncodeToBytes(block)
					if err != nil {
						return err
					}
					nativeBytes := snappy.Encode(blockBytes, blockBytes)
					nativeSize.Add(uint64(len(nativeBytes)))

					compactBlock := &CompactBlock{
						header:      block.Header(),
						uncles:      block.Uncles(),
						txHashes:    make([]common.Hash, 0),
						withdrawals: block.Withdrawals(),
					}
					for _, tx := range block.Transactions() {
						compactBlock.txHashes = append(compactBlock.txHashes, tx.Hash())
					}
					bcbBytes, err := rlp.EncodeToBytes(compactBlock)
					if err != nil {
						return err
					}
					bcbSnappyBytes := snappy.Encode(nil, bcbBytes)
					bcbSize.Add(uint64(len(bcbSnappyBytes)) * 2)
					for j := 0; j < 2; j++ {
						missTxs := make(types.Transactions, 0)
						for _, tx := range block.Transactions() {
							if !txpoolsList[j][height].Contains(tx.Hash()) {
								missTxs = append(missTxs, tx)
							}
						}
						bcbMissTxsBytes, err := rlp.EncodeToBytes(missTxs)
						if err != nil {
							return err
						}
						bcbMissTxsSnappyBytes := snappy.Encode(nil, bcbMissTxsBytes)
						bcbSize.Add(uint64(len(bcbMissTxsSnappyBytes)))
					}

					for j := 0; j < 2; j++ {
						missSize := 0
						hitFlag := true
						for _, tx := range block.Transactions() {
							if txpoolsList[j][height].Contains(tx.Hash()) {
								hitSize.Add(tx.Size())
							} else {
								hitFlag = false
								missSize += txSizeMap[tx.Hash()]
							}
						}
						if hitFlag {
							bcbHitNum.Add(1)
						}
						for k, rate := range rates {
							dataSize, paritySize := 0, 0
							for _, tx := range block.Transactions() {
								if !txpoolsList[j][height].Contains(tx.Hash()) {
									paritySize += txSizeMap[tx.Hash()]
								}
								dataSize += txSizeMap[tx.Hash()]
							}
							paritySize = int(float64(paritySize) * rate)
							chunkSize := max((dataSize+paritySize+65535)/65536, 1)
							chunkSize = (chunkSize + 63) / 64 * 64
							dataNum := (dataSize + chunkSize - 1) / chunkSize
							parityNum := (paritySize + chunkSize - 1) / chunkSize

							compactBlock.txSizes = make([]uint64, 0)
							compactBlock.chunkSize = uint64(chunkSize)
							compactBlock.parity = make([]byte, 0)

							if dataSize == 0 {
								eccbHitNum[k].Add(1)
								eccbBytes, err := rlp.EncodeToBytes(compactBlock)
								if err != nil {
									return err
								}
								eccbSnappyBytes := snappy.Encode(nil, eccbBytes)
								eccbSize[k].Add(uint64(len(eccbSnappyBytes)))
								continue
							}

							for _, tx := range block.Transactions() {
								compactBlock.txSizes = append(compactBlock.txSizes, uint64(txSizeMap[tx.Hash()]))
							}
							if dataNum > 0 && parityNum > 0 {
								chunkData := make([]byte, 0)
								for _, tx := range block.Transactions() {
									txBytes, err := rlp.EncodeToBytes(tx)
									if err != nil {
										panic(err)
									}
									txSnappyBytes := snappy.Encode(nil, txBytes)
									chunkData = append(chunkData, txSnappyBytes...)
								}
								chunkData = append(chunkData, make([]byte, (dataNum+parityNum)*chunkSize-len(chunkData))...)
								chunks := make([][]byte, dataNum+parityNum)
								for l := 0; l < dataNum+parityNum; l++ {
									chunks[l] = chunkData[l*chunkSize : (l+1)*chunkSize]
								}
								encoder, err := reedsolomon.New(dataNum, parityNum)
								if err != nil {
									panic(err)
								}
								if err = encoder.Encode(chunks); err != nil {
									panic(err)
								}
								compactBlock.parity = chunkData[dataNum*chunkSize:]
							}
							eccbBytes, err := rlp.EncodeToBytes(compactBlock)
							if err != nil {
								panic(err)
							}
							eccbSnappyBytes := snappy.Encode(nil, eccbBytes)
							eccbSize[k].Add(uint64(len(eccbSnappyBytes)))

							missChunks := mapset.NewThreadUnsafeSet[int]()
							offset := 0
							for _, tx := range block.Transactions() {
								if !txpoolsList[1-j][height].Contains(tx.Hash()) {
									for l := offset / chunkSize; l < (offset+txSizeMap[tx.Hash()]+chunkSize-1)/chunkSize; l++ {
										missChunks.Add(l)
									}
								}
								offset += txSizeMap[tx.Hash()]
							}
							if missChunks.Cardinality() <= parityNum {
								eccbHitNum[k].Add(1)
							}
						}
					}

					txTotalSize := uint64(0)
					txDataSize := uint64(0)
					for _, tx := range block.Transactions() {
						txTotalSize += tx.Size()
						txDataSize += uint64(len(tx.Data()))
					}
					aliasRate := max(1.0-float64(txTotalSize)*aliasChainRate/float64(txDataSize), 0)
					for _, tx := range block.Transactions() {
						value := reflect.ValueOf(tx).Elem()
						value = value.FieldByName("inner")
						value = reflect.NewAt(value.Type(), unsafe.Pointer(value.UnsafeAddr())).Elem()
						value = reflect.ValueOf(value.Interface()).Elem()
						value = value.FieldByName("Data")
						data := value.Interface().([]byte)
						data = data[:int(float64(len(data))*aliasRate)]
						value.Set(reflect.ValueOf(data))
					}
					aliasBytes, err := rlp.EncodeToBytes(block)
					if err != nil {
						return err
					}
					aliasBytes = snappy.Encode(aliasBytes, aliasBytes)
					aliasSize.Add(uint64(len(aliasBytes)))
				}
				return nil
			})
		}
		if err := wg.Wait(); err != nil {
			return err
		}

		if _, err := os.Stat("result"); errors.Is(err, os.ErrNotExist) {
			if err = os.Mkdir("result", 0755); err != nil {
				return err
			}
		}

		blockInfo := &BlockInfo{
			Heights:      interval,
			OriginalSize: totalSize.Load() / 2 / (interval[1] - interval[0]),
			NativeSize:   nativeSize.Load() / (interval[1] - interval[0]),
			AliasSize:    aliasSize.Load() / (interval[1] - interval[0]),
			BCBSize:      bcbSize.Load() / 2 / (interval[1] - interval[0]),
			BCBHitRate:   float64(bcbHitNum.Load()) / 2 / float64(interval[1]-interval[0]),
			ECCBSize:     make(map[string]uint64),
			ECCBHitRate:  make(map[string]float64),
			HitTxRate:    float64(hitSize.Load()) / float64(totalSize.Load()),
		}
		for j, rate := range rates {
			key := strconv.FormatFloat(rate, 'f', 2, 64)
			blockInfo.ECCBSize[key] = eccbSize[j].Load() / 2 / (interval[1] - interval[0])
			blockInfo.ECCBHitRate[key] = float64(eccbHitNum[j].Load()) / 2 / float64(interval[1]-interval[0])
		}
		blockInfoBytes, err := json.MarshalIndent(blockInfo, "", "    ")
		if err != nil {
			return err
		}
		if err = os.WriteFile(fmt.Sprintf("result/block-info-%d.json", i), blockInfoBytes, 0644); err != nil {
			return err
		}
	}

	return nil
}

type Peer struct {
	node    *Node
	latency float64
}

type Node struct {
	id    int
	peers []*Peer
}

type Graph []*Node

func parseGraphInfo(nodeNum int, beta float64, edgesInfo []byte) (Graph, error) {
	g := make(Graph, nodeNum)
	for i := 0; i < nodeNum; i++ {
		g[i] = &Node{
			id: i,
		}
	}
	edgesInfo = bytes.ReplaceAll(edgesInfo, []byte{'('}, []byte{'['})
	edgesInfo = bytes.ReplaceAll(edgesInfo, []byte{')'}, []byte{']'})
	var edges [][]int
	if err := json.Unmarshal(edgesInfo, &edges); err != nil {
		return nil, err
	}
	dists := make([]int, 0)
	for _, edge := range edges {
		selfId := edge[0]
		peerId := edge[1]
		dist := selfId - peerId
		if dist < 0 {
			dist = -dist
		}
		if dist > nodeNum-dist {
			dist = nodeNum - dist
		}
		dists = append(dists, dist)
	}
	slices.Sort(dists)
	boundDist := dists[int(float64(len(dists))*(1-beta))]
	for _, edge := range edges {
		u := edge[0]
		v := edge[1]
		dist := u - v
		if dist < 0 {
			dist = -dist
		}
		if dist > nodeNum-dist {
			dist = nodeNum - dist
		}
		var latency float64
		if dist < boundDist {
			latency = 50
		} else {
			latency = 1200
		}
		g[u].peers = append(g[u].peers, &Peer{
			node:    g[v],
			latency: latency,
		})
		g[v].peers = append(g[v].peers, &Peer{
			node:    g[u],
			latency: latency,
		})
	}
	return g, nil
}

type Protocol int

const (
	Native Protocol = iota
	AliasChain
	BCB
	ECCB
)

var protocols = []Protocol{Native, AliasChain, BCB, ECCB}

type Task struct {
	nodeNum    int
	peerNum    int
	beta       float64
	protocol   Protocol
	packetSize uint64
	bandwidth  uint64
	hitRate    float64
	source     int
}

func propagate(g Graph, task *Task) []float64 {
	r := rand.New(rand.NewPCG(uint64(time.Now().Unix()), uint64(time.Now().Unix())))
	froms := make([]int, len(g))
	result := make([]float64, len(g))
	for i := 0; i < len(result); i++ {
		froms[i] = -1
		result[i] = math.MaxInt
	}
	transLatency := float64(task.packetSize*1000) / float64(task.bandwidth)
	pq := prque.New[float64, int](nil)
	froms[task.source] = 0
	result[task.source] = 0
	pq.Push(task.source, 0)
	for !pq.Empty() {
		u, curTime := pq.Pop()
		curTime = -curTime
		degree := len(g[u].peers) - 1
		degreeSqrt := int(math.Sqrt(float64(degree)))
		count := 0
		for _, peer := range g[u].peers {
			v := peer.node.id
			if v == froms[u] {
				continue
			}
			propLatency := peer.latency
			var newTime float64
			switch task.protocol {
			case Native, AliasChain:
				if count%degreeSqrt == 0 {
					newTime = curTime + float64(degreeSqrt)*transLatency + propLatency
				} else {
					newTime = curTime + float64(degree)*transLatency + propLatency*5
				}
			case BCB, ECCB:
				if count%degreeSqrt == 0 {
					if r.Float64() < task.hitRate {
						newTime = curTime + float64(degreeSqrt)*transLatency + propLatency
					} else {
						newTime = curTime + float64(degreeSqrt)*transLatency + propLatency*3
					}
				} else {
					if r.Float64() < task.hitRate {
						newTime = curTime + float64(degree)*transLatency + propLatency*3
					} else {
						newTime = curTime + float64(degree)*transLatency + propLatency*5
					}
				}
			}
			newTime = curTime + (newTime-curTime)*(0.95+r.Float64()*0.1)
			if newTime < result[v] {
				froms[v] = u
				result[v] = newTime
				pq.Push(v, -newTime)
			}
			count++
		}
	}
	return result
}

func executeTask(task *Task) ([]float64, error) {
	nodeNum := task.nodeNum
	peerNum := task.peerNum
	beta := task.beta

	networkScript, err := os.CreateTemp("", "network-*.py")
	if err != nil {
		return nil, err
	}
	if _, err = networkScript.WriteString(networkScriptContent); err != nil {
		return nil, err
	}
	if err = networkScript.Close(); err != nil {
		return nil, err
	}
	defer os.Remove(networkScript.Name())

	cmd := exec.Command("python", networkScript.Name(), strconv.Itoa(nodeNum), strconv.Itoa(peerNum),
		strconv.FormatFloat(beta, 'f', -1, 64))
	graphInfoBytes, err := cmd.CombinedOutput()
	if err != nil {
		return nil, err
	}

	g, err := parseGraphInfo(nodeNum, beta, graphInfoBytes)
	if err != nil {
		panic(err)
	}

	latencies := propagate(g, task)
	slices.Sort(latencies)

	return latencies, nil
}

func parseBlockInfos(path string) ([]*BlockInfo, error) {
	blockInfos := make([]*BlockInfo, 0)
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if info.Mode()&os.ModeSymlink == os.ModeSymlink {
		path, err = os.Readlink(path)
		if err != nil {
			return nil, err
		}
	}
	if err := filepath.Walk(path, func(path string, info os.FileInfo, err error) error {
		if !strings.HasPrefix(info.Name(), "block-info") {
			return nil
		}
		fileBytes, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		blockInfo := new(BlockInfo)
		if err = json.Unmarshal(fileBytes, blockInfo); err != nil {
			return err
		}
		blockInfos = append(blockInfos, blockInfo)
		return nil
	}); err != nil {
		return nil, err
	}
	slices.SortFunc(blockInfos, func(a, b *BlockInfo) int {
		return cmp.Compare(a.Heights[0], b.Heights[0])
	})
	return blockInfos, nil
}

const aliasChainRate = 1 - (0.2823*0.4852 + 0.7177*0.6373)

func createGraphScript() (string, error) {
	graphScript, err := os.CreateTemp("", "graph-*.py")
	if err != nil {
		return "", err
	}
	if _, err = graphScript.WriteString(graphScriptContent); err != nil {
		return "", err
	}
	if err = graphScript.Close(); err != nil {
		return "", err
	}
	return graphScript.Name(), nil
}

func simulateCorrectionFactor(*cli.Context) error {
	const (
		nodeNum   = 6000
		peerNum   = 50
		beta      = 0.2
		bandwidth = 10 * 1024 * 1024 / 8
	)

	blockInfos, err := parseBlockInfos("result")
	if err != nil {
		return err
	}

	var wg errgroup.Group
	wg.SetLimit(runtime.NumCPU())
	var mu sync.Mutex
	result := make(map[*BlockInfo]map[string][]float64)
	for i := 0; i < 3; i++ {
		for _, blockInfo := range blockInfos {
			for rate := range blockInfo.ECCBSize {
				blockInfo, rate := blockInfo, rate
				wg.Go(func() error {
					latencies, err := executeTask(&Task{
						nodeNum:    nodeNum,
						peerNum:    peerNum,
						beta:       beta,
						protocol:   ECCB,
						packetSize: blockInfo.ECCBSize[rate],
						bandwidth:  bandwidth,
						hitRate:    blockInfo.ECCBHitRate[rate],
						source:     0,
					})
					if err != nil {
						return err
					}
					mu.Lock()
					if _, ok := result[blockInfo]; !ok {
						result[blockInfo] = make(map[string][]float64)
					}
					result[blockInfo][rate] = append(result[blockInfo][rate], latencies[int(float64(len(latencies))*0.99)])
					mu.Unlock()
					return nil
				})
			}
		}
	}
	if err = wg.Wait(); err != nil {
		return err
	}

	graphScript, err := createGraphScript()
	if err != nil {
		return err
	}
	defer os.Remove(graphScript)

	rates := []string{
		"1.00", "1.05", "1.10", "1.15", "1.16", "1.17", "1.18", "1.19",
		"1.20", "1.25", "1.30", "1.35", "1.40",
	}
	file, err := os.Create("result/simulate-correction-factor.csv")
	if err != nil {
		return err
	}
	writer := csv.NewWriter(file)
	for _, rate := range rates {
		record := []string{rate}
		for _, blockInfo := range blockInfos {
			for j := 0; j < 3; j++ {
				record = append(record, strconv.FormatFloat(1-blockInfo.ECCBHitRate[rate], 'f', -1, 64))
			}
		}
		for _, blockInfo := range blockInfos {
			for _, latency := range result[blockInfo][rate] {
				record = append(record, strconv.FormatFloat(latency, 'f', -1, 64))
			}
		}
		if err = writer.Write(record); err != nil {
			return err
		}
	}
	writer.Flush()
	if err = file.Close(); err != nil {
		return err
	}

	if _, err = os.Stat("images"); errors.Is(err, os.ErrNotExist) {
		if err = os.MkdirAll("images", 0755); err != nil {
			return err
		}
	}
	cmd := exec.Command("python", graphScript, "correction-factor", file.Name(),
		"images/simulate-correction-factor.pdf")
	if err = cmd.Run(); err != nil {
		return err
	}

	return nil
}

func simulateScalability(*cli.Context) error {
	const (
		peerNum   = 50
		beta      = 0.2
		bandwidth = 10 * 1024 * 1024 / 8
	)
	nodeNums := []int{1600, 16000, 160000}

	blockInfos, err := parseBlockInfos("result")
	if err != nil {
		return err
	}

	var wg errgroup.Group
	wg.SetLimit(runtime.NumCPU())
	var mu sync.Mutex
	result := make(map[int]map[*BlockInfo]map[Protocol][][]float64)
	for _, nodeNum := range nodeNums {
		for _, blockInfo := range blockInfos {
			for i := 0; i < 3; i++ {
				blockInfo, nodeNum := blockInfo, nodeNum
				wg.Go(func() error {
					latencies, err := executeTask(&Task{
						nodeNum:    nodeNum,
						peerNum:    peerNum,
						beta:       beta,
						protocol:   Native,
						packetSize: blockInfo.NativeSize,
						bandwidth:  bandwidth,
						hitRate:    0,
						source:     0,
					})
					if err != nil {
						return err
					}
					mu.Lock()
					if _, ok := result[nodeNum]; !ok {
						result[nodeNum] = make(map[*BlockInfo]map[Protocol][][]float64)
					}
					if _, ok := result[nodeNum][blockInfo]; !ok {
						result[nodeNum][blockInfo] = make(map[Protocol][][]float64)
					}
					result[nodeNum][blockInfo][Native] = append(result[nodeNum][blockInfo][Native], latencies)
					mu.Unlock()
					return nil
				})

				wg.Go(func() error {
					latencies, err := executeTask(&Task{
						nodeNum:    nodeNum,
						peerNum:    peerNum,
						beta:       beta,
						protocol:   AliasChain,
						packetSize: blockInfo.AliasSize,
						bandwidth:  bandwidth,
						hitRate:    0,
						source:     0,
					})
					if err != nil {
						return err
					}
					mu.Lock()
					if _, ok := result[nodeNum]; !ok {
						result[nodeNum] = make(map[*BlockInfo]map[Protocol][][]float64)
					}
					if _, ok := result[nodeNum][blockInfo]; !ok {
						result[nodeNum][blockInfo] = make(map[Protocol][][]float64)
					}
					result[nodeNum][blockInfo][AliasChain] = append(result[nodeNum][blockInfo][AliasChain], latencies)
					mu.Unlock()
					return nil
				})

				wg.Go(func() error {
					latencies, err := executeTask(&Task{
						nodeNum:    nodeNum,
						peerNum:    peerNum,
						beta:       beta,
						protocol:   BCB,
						packetSize: blockInfo.BCBSize,
						bandwidth:  bandwidth,
						hitRate:    blockInfo.BCBHitRate,
						source:     0,
					})
					if err != nil {
						panic(err)
					}
					mu.Lock()
					if _, ok := result[nodeNum]; !ok {
						result[nodeNum] = make(map[*BlockInfo]map[Protocol][][]float64)
					}
					if _, ok := result[nodeNum][blockInfo]; !ok {
						result[nodeNum][blockInfo] = make(map[Protocol][][]float64)
					}
					result[nodeNum][blockInfo][BCB] = append(result[nodeNum][blockInfo][BCB], latencies)
					mu.Unlock()
					return nil
				})

				rate := "1.18"
				wg.Go(func() error {
					latencies, err := executeTask(&Task{
						nodeNum:    nodeNum,
						peerNum:    peerNum,
						beta:       beta,
						protocol:   ECCB,
						packetSize: blockInfo.ECCBSize[rate],
						bandwidth:  bandwidth,
						hitRate:    blockInfo.ECCBHitRate[rate],
						source:     0,
					})
					if err != nil {
						panic(err)
					}
					mu.Lock()
					if _, ok := result[nodeNum]; !ok {
						result[nodeNum] = make(map[*BlockInfo]map[Protocol][][]float64)
					}
					if _, ok := result[nodeNum][blockInfo]; !ok {
						result[nodeNum][blockInfo] = make(map[Protocol][][]float64)
					}
					result[nodeNum][blockInfo][ECCB] = append(result[nodeNum][blockInfo][ECCB], latencies)
					mu.Unlock()
					return nil
				})
			}
		}
	}
	if err = wg.Wait(); err != nil {
		return err
	}

	graphScript, err := createGraphScript()
	if err != nil {
		return err
	}
	defer os.Remove(graphScript)

	for _, nodeNum := range nodeNums {
		file, err := os.Create(fmt.Sprintf("result/simulate-scalability-%d.csv", nodeNum))
		if err != nil {
			return err
		}
		writer := csv.NewWriter(file)
		for i := 0; i < nodeNum; i++ {
			record := make([]string, 0)
			for _, protocol := range []Protocol{Native, AliasChain, BCB, ECCB} {
				for _, blockInfo := range blockInfos {
					for _, latencies := range result[nodeNum][blockInfo][protocol] {
						record = append(record, strconv.FormatFloat(latencies[i], 'f', -1, 64))
					}
				}
			}
			if err = writer.Write(record); err != nil {
				return err
			}
		}
		writer.Flush()
		if err = file.Close(); err != nil {
			return err
		}

		if _, err = os.Stat("images"); errors.Is(err, os.ErrNotExist) {
			if err = os.MkdirAll("images", 0755); err != nil {
				return err
			}
		}
		cmd := exec.Command("python", graphScript, "scalability", file.Name(),
			fmt.Sprintf("images/simulate-scalability-%d.pdf", nodeNum))
		if err = cmd.Run(); err != nil {
			return err
		}
	}

	return nil
}

func analyzeScalability(*cli.Context) error {
	dirPath := "result"
	info, err := os.Lstat(dirPath)
	if err != nil {
		return err
	}
	if info.Mode()&os.ModeSymlink == os.ModeSymlink {
		dirPath, err = os.Readlink(dirPath)
		if err != nil {
			return err
		}
	}
	return filepath.Walk(dirPath, func(path string, info os.FileInfo, err error) error {
		var nodeNum int
		if _, err := fmt.Sscanf(info.Name(), "simulate-scalability-%d.csv", &nodeNum); err != nil {
			return nil
		}
		latencyMap := make(map[Protocol][]float64)

		file, err := os.Open(path)
		if err != nil {
			return err
		}
		defer file.Close()
		reader := csv.NewReader(file)
		for {
			records, err := reader.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				return err
			}
			for i, protocol := range protocols {
				num := len(records) / len(protocols)
				total := 0.0
				for j := 0; j < num; j++ {
					latency, err := strconv.ParseFloat(records[i*num+j], 64)
					if err != nil {
						return err
					}
					total += latency
				}
				latencyMap[protocol] = append(latencyMap[protocol], total/float64(num))
			}
		}

		fmt.Printf("%d", nodeNum)
		for _, protocol := range protocols {
			fmt.Printf(" %.2f", latencyMap[protocol][int(float64(len(latencyMap[protocol]))*0.99)])
		}
		fmt.Println()

		target := latencyMap[ECCB][int(float64(len(latencyMap[ECCB]))*0.99)]
		fmt.Printf("%d", nodeNum)
		for _, protocol := range protocols {
			latency := latencyMap[protocol][int(float64(len(latencyMap[protocol]))*0.99)]
			fmt.Printf(" %.2f", (latency-target)/latency*100)
		}
		fmt.Println()

		fmt.Printf("%d %.2f\n", nodeNum, target/1000)
		for _, protocol := range protocols {
			if protocol == ECCB {
				continue
			}
			index := sort.SearchFloat64s(latencyMap[protocol], target)
			if target-latencyMap[protocol][index-1] >= target-latencyMap[protocol][index] {
				index++
			}
			fmt.Printf("%d %d %.2f\n", nodeNum, protocol, float64(len(latencyMap[protocol]))*0.99/float64(index))
		}

		return err
	})
}

func simulateBlockSize(*cli.Context) error {
	const (
		nodeNum   = 6000
		peerNum   = 50
		beta      = 0.2
		bandwidth = 10 * 1024 * 1024 / 8
	)
	maxSizes := map[Protocol]uint64{
		Native:     500 * 1024,
		AliasChain: 700 * 1024,
		BCB:        2100 * 1024,
		ECCB:       2700 * 1024,
	}
	blockSizeMap := make(map[Protocol][]uint64)
	for _, protocol := range []Protocol{Native, AliasChain, BCB, ECCB} {
		for i := 0; i < 100; i++ {
			blockSizeMap[protocol] = append(blockSizeMap[protocol], (maxSizes[protocol]-1024)*uint64(i)/100+1024)
		}
	}

	blockInfos, err := parseBlockInfos("result")
	if err != nil {
		return err
	}

	var wg errgroup.Group
	wg.SetLimit(runtime.NumCPU())
	var mu sync.Mutex
	result := make(map[uint64]map[*BlockInfo]map[Protocol][]float64)
	for _, protocol := range []Protocol{Native, AliasChain, BCB, ECCB} {
		for _, blockSize := range blockSizeMap[protocol] {
			for _, blockInfo := range blockInfos {
				for i := 0; i < 3; i++ {
					blockInfo, blockSize := blockInfo, blockSize
					switch protocol {
					case Native:
						wg.Go(func() error {
							latencies, err := executeTask(&Task{
								nodeNum:    nodeNum,
								peerNum:    peerNum,
								beta:       beta,
								protocol:   Native,
								packetSize: blockSize * blockInfo.NativeSize / blockInfo.OriginalSize,
								bandwidth:  bandwidth,
								hitRate:    0,
								source:     0,
							})
							if err != nil {
								return err
							}
							mu.Lock()
							if _, ok := result[blockSize]; !ok {
								result[blockSize] = make(map[*BlockInfo]map[Protocol][]float64)
							}
							if _, ok := result[blockSize][blockInfo]; !ok {
								result[blockSize][blockInfo] = make(map[Protocol][]float64)
							}
							result[blockSize][blockInfo][Native] = append(result[blockSize][blockInfo][Native], latencies[int(float64(len(latencies))*0.99)])
							mu.Unlock()
							return nil
						})
					case AliasChain:
						wg.Go(func() error {
							latencies, err := executeTask(&Task{
								nodeNum:    nodeNum,
								peerNum:    peerNum,
								beta:       beta,
								protocol:   AliasChain,
								packetSize: blockSize * blockInfo.AliasSize / blockInfo.OriginalSize,
								bandwidth:  bandwidth,
								hitRate:    0,
								source:     0,
							})
							if err != nil {
								panic(err)
							}
							mu.Lock()
							if _, ok := result[blockSize]; !ok {
								result[blockSize] = make(map[*BlockInfo]map[Protocol][]float64)
							}
							if _, ok := result[blockSize][blockInfo]; !ok {
								result[blockSize][blockInfo] = make(map[Protocol][]float64)
							}
							result[blockSize][blockInfo][AliasChain] = append(result[blockSize][blockInfo][AliasChain], latencies[int(float64(len(latencies))*0.99)])
							mu.Unlock()
							return nil
						})
					case BCB:
						wg.Go(func() error {
							latencies, err := executeTask(&Task{
								nodeNum:    nodeNum,
								peerNum:    peerNum,
								beta:       beta,
								protocol:   BCB,
								packetSize: blockSize * blockInfo.BCBSize / blockInfo.OriginalSize,
								bandwidth:  bandwidth,
								hitRate:    blockInfo.BCBHitRate,
								source:     0,
							})
							if err != nil {
								panic(err)
							}
							mu.Lock()
							if _, ok := result[blockSize]; !ok {
								result[blockSize] = make(map[*BlockInfo]map[Protocol][]float64)
							}
							if _, ok := result[blockSize][blockInfo]; !ok {
								result[blockSize][blockInfo] = make(map[Protocol][]float64)
							}
							result[blockSize][blockInfo][BCB] = append(result[blockSize][blockInfo][BCB], latencies[int(float64(len(latencies))*0.99)])
							mu.Unlock()
							return nil
						})
					case ECCB:
						rate := "1.18"
						wg.Go(func() error {
							latencies, err := executeTask(&Task{
								nodeNum:    nodeNum,
								peerNum:    peerNum,
								beta:       beta,
								protocol:   ECCB,
								packetSize: blockSize * blockInfo.ECCBSize[rate] / blockInfo.OriginalSize,
								bandwidth:  bandwidth,
								hitRate:    blockInfo.ECCBHitRate[rate],
								source:     0,
							})
							if err != nil {
								return err
							}
							mu.Lock()
							if _, ok := result[blockSize]; !ok {
								result[blockSize] = make(map[*BlockInfo]map[Protocol][]float64)
							}
							if _, ok := result[blockSize][blockInfo]; !ok {
								result[blockSize][blockInfo] = make(map[Protocol][]float64)
							}
							result[blockSize][blockInfo][ECCB] = append(result[blockSize][blockInfo][ECCB], latencies[int(float64(len(latencies))*0.99)])
							mu.Unlock()
							return nil
						})
					}
				}
			}
		}
	}
	if err = wg.Wait(); err != nil {
		return err
	}

	graphScript, err := createGraphScript()
	if err != nil {
		return err
	}
	defer os.Remove(graphScript)

	file, err := os.Create("result/simulate-block-size.csv")
	if err != nil {
		return err
	}
	writer := csv.NewWriter(file)
	for i := 0; i < len(blockSizeMap[Native]); i++ {
		record := make([]string, 0)
		for _, protocol := range []Protocol{Native, AliasChain, BCB, ECCB} {
			blockSize := blockSizeMap[protocol][i]
			record = append(record, strconv.Itoa(int(blockSizeMap[protocol][i]/1024)))
			for _, blockInfo := range blockInfos {
				for _, latency := range result[blockSize][blockInfo][protocol] {
					record = append(record, strconv.FormatFloat(latency, 'f', -1, 64))
				}
			}
		}
		if err = writer.Write(record); err != nil {
			return err
		}
	}
	writer.Flush()
	if err = file.Close(); err != nil {
		return err
	}

	if _, err = os.Stat("images"); errors.Is(err, os.ErrNotExist) {
		if err = os.MkdirAll("images", 0755); err != nil {
			return err
		}
	}
	cmd := exec.Command("python", graphScript, "block-size", file.Name(), "images/simulate-block-size.pdf")
	return cmd.Run()
}

func analyzeBlockSize(*cli.Context) error {
	latencyMap := make(map[Protocol][][2]float64)

	file, err := os.Open("result/simulate-block-size.csv")
	if err != nil {
		return err
	}
	defer file.Close()
	reader := csv.NewReader(file)
	for {
		records, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		for i, protocol := range protocols {
			num := len(records) / len(protocols)
			total := 0.0
			for j := 1; j < num; j++ {
				latency, err := strconv.ParseFloat(records[i*num+j], 64)
				if err != nil {
					return err
				}
				total += latency
			}
			size, err := strconv.ParseFloat(records[i*num], 64)
			if err != nil {
				return err
			}
			latencyMap[protocol] = append(latencyMap[protocol], [2]float64{size, total / float64(num-1)})
		}
	}

	for _, protocol := range protocols {
		fmt.Printf("%.2f ", latencyMap[protocol][0][1])
	}
	fmt.Println()

	targetSizeMap := make(map[Protocol]float64)
	targetSizeMap[Native] = 140.0
	index := sort.Search(len(latencyMap[Native]), func(i int) bool {
		return latencyMap[Native][i][0] >= targetSizeMap[Native]
	})
	targetLatency := (latencyMap[Native][index][1]-latencyMap[Native][index-1][1])/(latencyMap[Native][index][0]-latencyMap[Native][index-1][0])*(targetSizeMap[Native]-latencyMap[Native][index-1][0]) + latencyMap[Native][index-1][1]
	for _, protocol := range protocols {
		index = sort.Search(len(latencyMap[protocol]), func(i int) bool {
			return latencyMap[protocol][i][1] >= targetLatency
		})
		targetSizeMap[protocol] = (latencyMap[protocol][index][0]-latencyMap[protocol][index-1][0])/(latencyMap[protocol][index][1]-latencyMap[protocol][index-1][1])*(targetLatency-latencyMap[protocol][index-1][1]) + latencyMap[protocol][index-1][0]
	}
	fmt.Println(targetLatency)
	for _, protocol := range protocols {
		fmt.Printf("%.2f ", targetSizeMap[protocol])
	}
	fmt.Println()
	for _, protocol := range protocols {
		fmt.Printf("%.2f ", targetSizeMap[ECCB]/targetSizeMap[protocol])
	}
	fmt.Println()

	targetLatency = 12000.0
	for _, protocol := range protocols {
		index = sort.Search(len(latencyMap[protocol]), func(i int) bool {
			return latencyMap[protocol][i][1] >= targetLatency
		})
		targetSizeMap[protocol] = (latencyMap[protocol][index][0]-latencyMap[protocol][index-1][0])/(latencyMap[protocol][index][1]-latencyMap[protocol][index-1][1])*(targetLatency-latencyMap[protocol][index-1][1]) + latencyMap[protocol][index-1][0]
	}
	fmt.Println(targetLatency)
	for _, protocol := range protocols {
		fmt.Printf("%.2f ", targetSizeMap[protocol])
	}
	fmt.Println()
	for _, protocol := range protocols {
		fmt.Printf("%.2f ", targetSizeMap[ECCB]/targetSizeMap[protocol])
	}
	fmt.Println()

	return err
}

func simulateBandwidth(*cli.Context) error {
	const (
		nodeNum = 6000
		peerNum = 50
		beta    = 0.2
	)
	bandwidths := []uint64{10 * 1024 * 1024 / 8, 20 * 1024 * 1024 / 8, 40 * 1024 * 1024 / 8}

	blockInfos, err := parseBlockInfos("result")
	if err != nil {
		return err
	}

	var wg errgroup.Group
	wg.SetLimit(runtime.NumCPU())
	var mu sync.Mutex
	result := make(map[uint64]map[*BlockInfo]map[Protocol][][]float64)
	for _, bandwidth := range bandwidths {
		for _, blockInfo := range blockInfos {
			blockInfo, bandwidth := blockInfo, bandwidth
			for i := 0; i < 3; i++ {
				wg.Go(func() error {
					latencies, err := executeTask(&Task{
						nodeNum:    nodeNum,
						peerNum:    peerNum,
						beta:       beta,
						protocol:   Native,
						packetSize: blockInfo.NativeSize,
						bandwidth:  bandwidth,
						hitRate:    0,
						source:     0,
					})
					if err != nil {
						return err
					}
					mu.Lock()
					if _, ok := result[bandwidth]; !ok {
						result[bandwidth] = make(map[*BlockInfo]map[Protocol][][]float64)
					}
					if _, ok := result[bandwidth][blockInfo]; !ok {
						result[bandwidth][blockInfo] = make(map[Protocol][][]float64)
					}
					result[bandwidth][blockInfo][Native] = append(result[bandwidth][blockInfo][Native], latencies)
					mu.Unlock()
					return nil
				})

				wg.Go(func() error {
					latencies, err := executeTask(&Task{
						nodeNum:    nodeNum,
						peerNum:    peerNum,
						beta:       beta,
						protocol:   AliasChain,
						packetSize: blockInfo.AliasSize,
						bandwidth:  bandwidth,
						hitRate:    0,
						source:     0,
					})
					if err != nil {
						return err
					}
					mu.Lock()
					if _, ok := result[bandwidth]; !ok {
						result[bandwidth] = make(map[*BlockInfo]map[Protocol][][]float64)
					}
					if _, ok := result[bandwidth][blockInfo]; !ok {
						result[bandwidth][blockInfo] = make(map[Protocol][][]float64)
					}
					result[bandwidth][blockInfo][AliasChain] = append(result[bandwidth][blockInfo][AliasChain], latencies)
					mu.Unlock()
					return nil
				})

				wg.Go(func() error {
					latencies, err := executeTask(&Task{
						nodeNum:    nodeNum,
						peerNum:    peerNum,
						beta:       beta,
						protocol:   BCB,
						packetSize: blockInfo.BCBSize,
						bandwidth:  bandwidth,
						hitRate:    blockInfo.BCBHitRate,
						source:     0,
					})
					if err != nil {
						return err
					}
					mu.Lock()
					if _, ok := result[bandwidth]; !ok {
						result[bandwidth] = make(map[*BlockInfo]map[Protocol][][]float64)
					}
					if _, ok := result[bandwidth][blockInfo]; !ok {
						result[bandwidth][blockInfo] = make(map[Protocol][][]float64)
					}
					result[bandwidth][blockInfo][BCB] = append(result[bandwidth][blockInfo][BCB], latencies)
					mu.Unlock()
					return nil
				})

				rate := "1.18"
				wg.Go(func() error {
					latencies, err := executeTask(&Task{
						nodeNum:    nodeNum,
						peerNum:    peerNum,
						beta:       beta,
						protocol:   ECCB,
						packetSize: blockInfo.ECCBSize[rate],
						bandwidth:  bandwidth,
						hitRate:    blockInfo.ECCBHitRate[rate],
						source:     0,
					})
					if err != nil {
						panic(err)
					}
					mu.Lock()
					if _, ok := result[bandwidth]; !ok {
						result[bandwidth] = make(map[*BlockInfo]map[Protocol][][]float64)
					}
					if _, ok := result[bandwidth][blockInfo]; !ok {
						result[bandwidth][blockInfo] = make(map[Protocol][][]float64)
					}
					result[bandwidth][blockInfo][ECCB] = append(result[bandwidth][blockInfo][ECCB], latencies)
					mu.Unlock()
					return nil
				})
			}
		}
	}
	if err = wg.Wait(); err != nil {
		return err
	}

	graphScript, err := createGraphScript()
	if err != nil {
		return err
	}
	defer os.Remove(graphScript)

	for _, bandwidth := range bandwidths {
		file, err := os.Create(fmt.Sprintf("result/simulate-bandwidth-%d.csv", bandwidth*8/1024/1024))
		if err != nil {
			return err
		}
		writer := csv.NewWriter(file)
		for i := 0; i < nodeNum; i++ {
			record := make([]string, 0)
			for _, protocol := range []Protocol{Native, AliasChain, BCB, ECCB} {
				for _, blockInfo := range blockInfos {
					for _, latencies := range result[bandwidth][blockInfo][protocol] {
						record = append(record, strconv.FormatFloat(latencies[i], 'f', -1, 64))
					}
				}
			}
			if err = writer.Write(record); err != nil {
				return err
			}
		}
		writer.Flush()
		if err = file.Close(); err != nil {
			return err
		}

		if _, err = os.Stat("images"); errors.Is(err, os.ErrNotExist) {
			if err = os.MkdirAll("images", 0755); err != nil {
				return err
			}
		}
		cmd := exec.Command("python", graphScript, "bandwidth", file.Name(),
			fmt.Sprintf("images/simulate-bandwidth-%d.pdf", bandwidth*8/1024/1024))
		if err = cmd.Run(); err != nil {
			return err
		}
	}

	return nil
}

func analyzeBandwidth(*cli.Context) error {
	dirPath := "result"
	info, err := os.Lstat(dirPath)
	if err != nil {
		return err
	}
	if info.Mode()&os.ModeSymlink == os.ModeSymlink {
		dirPath, err = os.Readlink(dirPath)
		if err != nil {
			return err
		}
	}
	bandwidths := make([]int, 0)
	latencyMap := make(map[int]map[Protocol][]float64)
	if err = filepath.Walk(dirPath, func(path string, info os.FileInfo, err error) error {
		var bandwidth int
		if _, err := fmt.Sscanf(info.Name(), "simulate-bandwidth-%d.csv", &bandwidth); err != nil {
			return nil
		}
		bandwidths = append(bandwidths, bandwidth)

		file, err := os.Open(path)
		if err != nil {
			return err
		}
		defer file.Close()
		reader := csv.NewReader(file)
		for {
			records, err := reader.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				return err
			}
			for i, protocol := range protocols {
				num := len(records) / len(protocols)
				total := 0.0
				for j := 0; j < num; j++ {
					latency, err := strconv.ParseFloat(records[i*num+j], 64)
					if err != nil {
						return err
					}
					total += latency
				}
				if _, ok := latencyMap[bandwidth]; !ok {
					latencyMap[bandwidth] = make(map[Protocol][]float64)
				}
				latencyMap[bandwidth][protocol] = append(latencyMap[bandwidth][protocol], total/float64(num))
			}
		}
		return err
	}); err != nil {
		return err
	}
	slices.Sort(bandwidths)

	for _, bandwidth := range bandwidths {
		for _, protocol := range protocols {
			fmt.Printf("%.2f ", 100-latencyMap[bandwidth][protocol][int(float64(len(latencyMap[bandwidth][protocol]))*0.99)]/latencyMap[bandwidths[0]][protocol][int(float64(len(latencyMap[bandwidths[0]][protocol]))*0.99)]*100)
		}
		fmt.Println()
	}

	return nil
}

func simulateSimilarity(*cli.Context) error {
	const (
		nodeNum   = 6000
		peerNum   = 50
		beta      = 0.2
		bandwidth = 10 * 1024 * 1024 / 8
	)
	type Pair struct {
		protocol   Protocol
		similarity int
	}
	pairs := []Pair{{Native, 0}, {AliasChain, 0}}
	for _, protocol := range []Protocol{BCB, ECCB} {
		for _, similarity := range []int{60, 70, 80, 90} {
			pairs = append(pairs, Pair{protocol, similarity})
		}
	}

	blockInfos, err := parseBlockInfos("result")
	if err != nil {
		return err
	}

	var wg errgroup.Group
	wg.SetLimit(runtime.NumCPU())
	var mu sync.Mutex
	result := make(map[*BlockInfo]map[Pair][][]float64)
	for _, blockInfo := range blockInfos {
		for i := 0; i < 3; i++ {
			blockInfo, rate := blockInfo, "1.18"
			wg.Go(func() error {
				latencies, err := executeTask(&Task{
					nodeNum:    nodeNum,
					peerNum:    peerNum,
					beta:       beta,
					protocol:   Native,
					packetSize: blockInfo.NativeSize,
					bandwidth:  bandwidth,
					hitRate:    0,
					source:     0,
				})
				if err != nil {
					return err
				}
				mu.Lock()
				if _, ok := result[blockInfo]; !ok {
					result[blockInfo] = make(map[Pair][][]float64)
				}
				result[blockInfo][Pair{Native, 0}] = append(result[blockInfo][Pair{Native, 0}], latencies)
				mu.Unlock()
				return nil
			})

			wg.Go(func() error {
				latencies, err := executeTask(&Task{
					nodeNum:    nodeNum,
					peerNum:    peerNum,
					beta:       beta,
					protocol:   AliasChain,
					packetSize: blockInfo.AliasSize,
					bandwidth:  bandwidth,
					hitRate:    0,
					source:     0,
				})
				if err != nil {
					return err
				}
				mu.Lock()
				if _, ok := result[blockInfo]; !ok {
					result[blockInfo] = make(map[Pair][][]float64)
				}
				result[blockInfo][Pair{AliasChain, 0}] = append(result[blockInfo][Pair{AliasChain, 0}], latencies)
				mu.Unlock()
				return nil
			})

			for _, similarity := range []int{60, 70, 80, 90} {
				similarity := similarity
				wg.Go(func() error {
					latencies, err := executeTask(&Task{
						nodeNum:    nodeNum,
						peerNum:    peerNum,
						beta:       beta,
						protocol:   BCB,
						packetSize: uint64(float64(blockInfo.BCBSize*uint64(100-similarity)) / 100 / (1 - blockInfo.HitTxRate)),
						bandwidth:  bandwidth,
						hitRate:    blockInfo.BCBHitRate,
						source:     0,
					})
					if err != nil {
						return err
					}
					mu.Lock()
					if _, ok := result[blockInfo]; !ok {
						result[blockInfo] = make(map[Pair][][]float64)
					}
					result[blockInfo][Pair{BCB, similarity}] = append(result[blockInfo][Pair{BCB, similarity}], latencies)
					mu.Unlock()
					return nil
				})

				wg.Go(func() error {
					latencies, err := executeTask(&Task{
						nodeNum:    nodeNum,
						peerNum:    peerNum,
						beta:       beta,
						protocol:   ECCB,
						packetSize: uint64(float64(blockInfo.ECCBSize[rate]*uint64(100-similarity)) / 100 / (1 - blockInfo.HitTxRate)),
						bandwidth:  bandwidth,
						hitRate:    blockInfo.ECCBHitRate[rate],
						source:     0,
					})
					if err != nil {
						panic(err)
					}
					mu.Lock()
					if _, ok := result[blockInfo]; !ok {
						result[blockInfo] = make(map[Pair][][]float64)
					}
					result[blockInfo][Pair{ECCB, similarity}] = append(result[blockInfo][Pair{ECCB, similarity}], latencies)
					mu.Unlock()
					return nil
				})
			}
		}
	}
	if err = wg.Wait(); err != nil {
		return err
	}

	graphScript, err := createGraphScript()
	if err != nil {
		return err
	}
	defer os.Remove(graphScript)

	file, err := os.Create("result/simulate-similarity.csv")
	if err != nil {
		return err
	}
	writer := csv.NewWriter(file)
	for i := 0; i < nodeNum; i++ {
		record := make([]string, 0)
		for _, pair := range pairs {
			for _, blockInfo := range blockInfos {
				for _, latencies := range result[blockInfo][pair] {
					record = append(record, strconv.FormatFloat(latencies[i], 'f', -1, 64))
				}
			}
		}
		if err = writer.Write(record); err != nil {
			return err
		}
	}
	writer.Flush()
	if err = file.Close(); err != nil {
		return err
	}

	if _, err = os.Stat("images"); errors.Is(err, os.ErrNotExist) {
		if err = os.MkdirAll("images", 0755); err != nil {
			return err
		}
	}
	cmd := exec.Command("python", graphScript, "similarity", file.Name(), "images/simulate-similarity.pdf")
	return cmd.Run()
}

func analyzeSimilarity(*cli.Context) error {
	type Pair struct {
		protocol   Protocol
		similarity int
	}
	pairs := []Pair{{Native, 0}, {AliasChain, 0}}
	for _, protocol := range []Protocol{BCB, ECCB} {
		for _, similarity := range []int{60, 70, 80, 90} {
			pairs = append(pairs, Pair{protocol, similarity})
		}
	}

	latencyMap := make(map[Pair][]float64)

	file, err := os.Open("result/simulate-similarity.csv")
	if err != nil {
		return err
	}
	defer file.Close()
	reader := csv.NewReader(file)
	for {
		records, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		for i, pair := range pairs {
			latencyMap[pair] = append(latencyMap[pair], 0)
			num := len(records) / len(pairs)
			for j := 0; j < num; j++ {
				latency, err := strconv.ParseFloat(records[i*num+j], 64)
				if err != nil {
					return err
				}
				latencyMap[pair][len(latencyMap[pair])-1] += latency
			}
			latencyMap[pair][len(latencyMap[pair])-1] /= float64(num)
		}
	}

	for _, pair := range pairs {
		fmt.Println(pair, latencyMap[pair][int(float64(len(latencyMap[pair]))*0.99)])
	}

	return nil
}

func main() {
	if err := app.Run(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
