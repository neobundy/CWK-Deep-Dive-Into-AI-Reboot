# Makefile
METAL   = xcrun -sdk macosx metal
SWIFTC  = xcrun -sdk macosx swiftc

all: hello_host

hello.metallib: hello.metal
	$(METAL) $< -o $@

hello_host: hello_host.swift hello.metallib
	$(SWIFTC) $< -framework Metal -framework Foundation -o $@

run: hello_host
	./hello_host

clean:
	rm -f hello.metallib hello_host