BINARY      := whisper-oai
BIN_DIR     := bin
PKG         := ./...
VERSION     ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS     := -s -w -X main.version=$(VERSION)

.PHONY: all build run test vet fmt tidy clean install

all: build

## build: compile the binary into bin/
build:
	@mkdir -p $(BIN_DIR)
	CGO_ENABLED=0 go build -trimpath -ldflags "$(LDFLAGS)" -o $(BIN_DIR)/$(BINARY) ./cmd/whisper-oai

## run: run the server with the example config
run: build
	$(BIN_DIR)/$(BINARY) serve --config config/whisper-oai.example.yaml

## test: run unit tests
test:
	go test $(PKG) -race -count=1

## vet: static analysis
vet:
	go vet $(PKG)

## fmt: format code
fmt:
	gofmt -s -w .

## tidy: sync go.mod / go.sum
tidy:
	go mod tidy

## install: install to $GOPATH/bin
install:
	go install -ldflags "$(LDFLAGS)" ./cmd/whisper-oai

## clean: remove build artifacts
clean:
	rm -rf $(BIN_DIR) coverage.out
