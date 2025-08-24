.PHONY: bnsim clean

bnsim:
	go build -v .

clean:
	rm bnsim
	go clean -cache
