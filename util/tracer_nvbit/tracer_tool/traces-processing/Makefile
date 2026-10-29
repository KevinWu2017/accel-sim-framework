TARGET := post-traces-processing

$(TARGET): post-traces-processing.cpp weight_address_map.h
	g++ -std=c++17 -O3 -g -pthread -o $@ post-traces-processing.cpp -lzstd

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o
