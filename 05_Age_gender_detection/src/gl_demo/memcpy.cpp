#include <stdio.h>
#include "chrono"
#include <stdlib.h>
#include <cstring>
#include <iostream>
#include <array>
static std::array<char, 1920*1080*3> testBuffer;
static std::array<char, 1920*1080*3> testBuffer4;
int main()
{
    memset(testBuffer4.data(),0x0,testBuffer4.size());
    memset(testBuffer.data(),0xFF,testBuffer4.size());
    uint64_t start = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    memcpy(testBuffer4.data(),testBuffer.data(),testBuffer4.size());
    uint64_t end = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    auto time_difference = end - start;
 
    
    std::cout << "Memcpy took " << time_difference << std::endl;
    std::cout << "Value is 0x" << testBuffer[0] << std::endl;

}