#pragma once

#include <string>

using namespace std;

const char* cgenerate_random1[ ] = // should be done on the CPU
{
"__kernel void generate_random1(__global unsigned long p_current_time, __global unsigned long* p_ulong_rand)", // Line 104
"{",
"	unsigned int gid = get_global_id(0);", //,count = 0;",
"	unsigned long width = 500, height = 500;",
"	bool loop_condition = true;",
"	do",
"	{",
"		unsigned long seed = p_current_time + gid;", // From:
"		seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);",
"		unsigned long result = (seed >> 16) % (width * height);", // or seed >> 16;

"			p_ulong_rand[gid] = result;",
"			loop_condition = false;",
"	}",
"	while(loop_condition);",
"   barrier(CLK_GLOBAL_MEM_FENCE);",
"}"
}; // Line 18
const char* cgenerate_random2[] =
{
"kernel void generate_random(__global int* random, __global int* out) {",
"     int gid = get_global_id(0);",
"    out[gid] = gid* (*random) % 5;",
"}",
};
// const size_t generate_random_t = generate_random_ulong_kernel.length();
const char* generate_next_matrix_kernel[ ] =
{
"__kernel void generate_next_matrix(",
"__global const int matrix[HEIGHT*WIDTH],",
" __global int buffer_matrix[HEIGHT*WIDTH] )",
"{",
"	int gid = get_global_id(0); ",
"	buffer_matrix[gid] = (matrix[gid]+1) % 4;",
"}"
};
//const char* char_generate_random = { (generate_random_kernel.c_str()) };
//const char* char_generate_next_matrix = { (generate_next_matrix_kernel.c_str()) };