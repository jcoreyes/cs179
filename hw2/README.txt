CS 179 Set 2
Due Wednesday 4/15/2015 @ 3PM.

Put all answers in a file called README.txt.
After answering all of the questions, list how long part 1 and part 2 took.
Feel free to leave any other feedback.

Submit completed sets by emailing to emartin@caltech.edu with subject
"CS 179 Set 2 Submission - Name" where "Name" is your name. Attach to the email
a single archive file (.zip, .tar, .gz, .tar.gz/.tgz) with your README file and
all code.

PART 1
Question 1.1: Latency Hiding (5 points)
---------------------------------------
Approximately how many arithmetic instructions does it take to hide the latency
of a single arithmetic instruction on a GK110?
Assume all of the arithmetic instructions are independent (ie have no
instruction dependencies).
You do not need to consider the number of execution cores on the chip.

Hint: What is the latency of an arithmetic instruction? How many instructions
can a GK110 begin issuing in 1 clock cycle (assuming no dependencies)?

The latency of an arithmetic instruction is about 10 ns. A GK110 has 4 warp
schedulers with 2 dispatchers each. It can start instructions in up to 4 warps
each clock and up to 2 instructions in each warp. So it can start 8 instructions
in 1 clock cycle assuming no dependencies. One arithmetic instruction can be done
in 10 clocks so it'll take about 10 * 8 = 80 arithmetic
instructions to hide the latency of a single arithmetic instruction.



Question 1.2: Thread Divergence (6 points)
------------------------------------------
Let the block shape be (32, 32, 1).

(a)
int idx = threadIdx.y + blockSize.y * threadIdx.x;
if (idx % 32 < 16) {
    foo();
} else {
    bar();
}

Does this code diverge? Why or why not?
This code does not diverge. In a given warp, threadIdx.y will be the same
and thread Idx.x will range from 0 to 31. So the idx's of all threads
in a warp will be 32 apart from each other. So idx mod 32 will evaluate
to the same number for all threads in a warp. Thus all threads in a warp
will either perform foo() or all threads will perform bar(). Since no
threads in a warp are waiting and doing nothing we do not have divergence.


(b)
const float pi = 3.14;
float result = 1.0;
for (int i = 0; i < threadIdx.x; i++) {
    result *= pi;
}

Does this code diverge? Why or why not?
(This is a bit of a trick question, either "yes" or "no can be a correct answer
with appropriate explanation).
This code diverges. In a given warp, threadIdx.x will vary from 0 to 31. This means
that the code loops a different number of times for each thread in the warp. So 
threads are disabled as their iteration counts reach their set numbers but the warp
as a whole must keep looping until the thread with the highest iteration count is
done. So while the thread with threadIdx.x=31 is still looping the other threads
are doing nothing. Thus the code diverges.

Question 1.3: Coalesced Memory Access (9 points)
------------------------------------------------
Let the block shape be (32, 32, 1).
Let data be a (float *) pointing to global memory and let data be 128 byte
aligned (so data % 128 == 0).

Consider each of the following access patterns.

(a)
data[threadIdx.x + blockSize.x * threadIdx.y] = 1.0;

Is this write coalesced? How many 128 byte cache lines does this write to?

This write is coalesced. In a given warp, threadIdx.x will vary from 0 to
31 so the write will access 32 consecutive floats offset starting from 0.
A float is 4 bytes 32 floats * 4 bytes = 128 bytes so each warp writes
to one 128 byte cache line.


(b)
data[threadIdx.y + blockSize.y * threadIdx.x] = 1.0;

Is this write coalesced? How many 128 byte cache lines does this write to?

This write is not coalesced because each warp is not accessing the minimum
possible 128 byte cache lines. The code does not access the array consecutively.
In a given warp, threadIdx.y will be the same and threadIdx.x will vary from 0 to
31. So we can think of a warp as accessing 32 consecutive rows 1 column per instruction
if the data array is 2d where each access is 32 floats = 128 bytes away from the previous
access.
So each warp will access 32 different 128 byte cache lines since in a warp
each thread is accessing a float 128 bytes away from each the next closest access.

(c)
data[1 + threadIdx.x + blockSize.x * threadIdx.y] = 1.0;

Is this write coalesced? How many 128 byte cache lines does this write to?
This write is not coalesced because each warp is not accessing the minimum
possible number of 128 byte cache lines. Since the data array is 128 byte
aligned (so data % 128 == 0), then each warp will access 2 different 128 byte
cache lines. Instead of one warp accessing 4 byte data from addresss 0 to 124
the warp will access addresses from 4 to 128. So it must access 2 different
128 byte cache lines.

Question 1.4: Bank Conflicts and Instruction Dependencies (15 points)
---------------------------------------------------------------------
Let's consider multiplying a 32 x 128 matrix with a 128 x 32
element matrix. This outputs a 32 x 32 matrix. We'll use 32 ** 2 = 1024 threads
and each thread will compute 1 output element.
Although its not optimal, for the sake of simplicity let's use a single block,
so grid shape = (1, 1, 1), block shape = (32, 32, 1).

For the sake of this problem, let's assume both the left and right matrices have
already been stored in shared memory are in column major format. This means
element in the ith row and jth column is accessible at lhs[i + 32 * j] for the
left hand side and rhs[i + 128 * j] for the right hand side.

This kernel will write to a variable called output stored in shared memory.

Consider the following kernel code:

int i = threadIdx.x;
int j = threadIdx.y;
for (int k = 0; k < 128; k += 2) {
    output[i + 32 * j] += lhs[i + 32 * k] * rhs[k + 128 * j];
    output[i + 32 * j] += lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];
}

(a)
Are there bank conflicts in this code? If so, how many ways is the bank conflict
(2-way, 4-way, etc)?
There are no bank conflicts in this code. In a given warp, i, will vary from
0 to 31 and j wil be fixed.. We can think of k as fixed since we're considering 1 
instruction at a time in a warp. In the first line of the for loop, for lhs[i+32*k],
each thread in a warp will access 32 different banks. For rhs[k+128*j] each thread
in a warp will access the same memory address from 1 bank. The same applies for
the second line of the for loop. Thus there are no bank conflicts because we do
not have multiple threads in the same warp accessing different addresses from
the same bank.

(b)
Expand the inner part of the loop (below)

output[i + 32 * j] += lhs[i + 32 * k] * rhs[k + 128 * j];
output[i + 32 * j] += lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];

into "psuedo-assembly" as was done in the coordinate addition example in lecture
4.

There's no need to expand the indexing math, only to expand the loads, stores,
and math. Notably, the operation a += b * c can be computed by a single
instruction called a fused multiply add (FMA), so this can be a single
instruction in your "psuedo-assembly".

Hint: Each line should expand to 5 instructions.
1. x0 = lhs[i+32*k];
2. y0 = rhs[k+128*j];
3. w0 = output[i+32*j];
4. w0 = w0 + x0 * y0; // FMA
5. output[i+32*j] = w0

6. x1 = lhs[i + 32 * (k + 1)];
7. y1 = rhs[(k + 1) + 128 * j];
8. w1 = output[i+32*j];
9. w1 = w1 + x1 * y1; // FMA
10. output[i+32*j] = w1;

(c)
Identify pairs of dependent instructions in your answer to part b.
Line 4 depends on the previous three lines.
Line 5 depends on line 4.
Line 8 depends on line 5. 
Line 9 depends on the previous three lines.
Line 10 depends on line 9.

(d)
Rewrite the code given at the beginning of this problem to minimize instruction
dependencies. You can add or delete instructions (deleting an instruction is a
valid way to get rid of a dependency!) but each iteration of the loop must still
process 2 values of k.
We can rewrite the code so that the second line in the for loop does not 
depend on the first line in the for loop so that we do the multiplications
independently and add them together at the end. So the code becomes
int i = threadIdx.x;
int j = threadIdx.y;
float a, b;
for (int k = 0; k < 128; k += 2) {
	a = lhs[i + 32 * k] * rhs[k + 128 * j];
	b = lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];
	output[i + 32 * j] += a + b
}
Now (in the for loop) the second line does not depend on the first line. 
We just have the third line depending on the two previous lines but the 
two previous lines are the more computationally intensive lines so we want 
them to happen in parallel.

(e)
Can you think of any other anything else you can do that might make this code
run faster?
If we have enough register then we want to do the loads all at the same time and
overlap all the data io. So the code might become:
int i = threadIdx.x;
int j = threadIdx.y;
float a, b, c, d, e, f;
for (int k = 0; k < 128; k += 2) {
	c = lhs[i + 32 * k];
	d = rhs[k + 128 * j];
	e = lhs[i + 32 * (k + 1)];
	f = rhs[(k + 1) + 128 * j];
	a = c * d;
	b = e * f;
	output[i + 32 * j] += a + b
}
We can also unroll the for loop to avoid instruction dependencies.
================================================================================

PART 2 - Matrix transpose optimization (65 points)
Optimize the CUDA matrix transpose implementations in transpose_cuda.cu.
Read ALL of the TODO comments. Matrix transpose is a common exercise in GPU
optimization, so do not search for existing GPU matrix transpose code on the
internet.

Your transpose code only need to be able to transpose square matrices where
the side length is a multiple of 64.

The initial implementation has each block of 1024 threads handle a 64x64
block of the matrix, but you can change anything about the kernel if it helps
obtain better performance.

The main method of transpose.cc already checks for correctness for all transpose
results, so there should be an assertion failure if your kernel produces incorrect
output.

The purpose of the shmemTransposeKernel is to demonstrate proper usage of
global and shared memory. The optimalTransposeKernel should be built on top of
shmemTransposeKernel and should incorporate any "tricks" such as ILP, loop unrolling,
vectorized IO, etc that have been discussed in class.

You can compile and run the code by running

make transpose
./transpose

and the build process was tested on minuteman. If this does not work on minutman
for you, be sure to add the lines

export PATH=/usr/local/cuda-6.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-6.5/lib64:$LD_LIBRARY_PATH

to your ~/.profile file (and then exit and ssh back in to restart your shell).

The transpose program takes 2 optional arguments: input size and method.
Input size must be one of -1, 512, 1024, 2048, 4096, and method must be one
all, cpu, gpu_memcpy, naive, shmem, optimal.
Input size is the first argument and defaults to -1. Method is the second
argument and defaults to all. You can pass input size without passing method,
but you cannot pass method without passing an input size.

Examples:
./transpose
./transpose 512
./transpose 4096 naive
./transpose -1 optimal

Copy paste the output of ./transpose.cc into README.txt once you are done.
Describe the strategies used for performance in either block comments over the
kernel (as done for naiveTransposeKernel) or in README.txt.

Size 512 naive CPU: 0.277920 ms
Size 512 GPU memcpy: 0.032608 ms
Size 512 naive GPU: 0.094784 ms
Size 512 shmem GPU: 0.027008 ms
Size 512 optimal GPU: 0.024512 ms

Size 1024 naive CPU: 2.368256 ms
Size 1024 GPU memcpy: 0.081280 ms
Size 1024 naive GPU: 0.309888 ms
Size 1024 shmem GPU: 0.091104 ms
Size 1024 optimal GPU: 0.085760 ms

Size 2048 naive CPU: 34.122913 ms
Size 2048 GPU memcpy: 0.262752 ms
Size 2048 naive GPU: 1.165184 ms
Size 2048 shmem GPU: 0.336960 ms
Size 2048 optimal GPU: 0.304064 ms

Size 4096 naive CPU: 155.034363 ms
Size 4096 GPU memcpy: 1.000896 ms
Size 4096 naive GPU: 4.108256 ms
Size 4096 shmem GPU: 1.218432 ms
Size 4096 optimal GPU: 1.168288 ms

================================================================================

BONUS (+5 points, maximum set score is 100 even with bonus)

Mathematical scripting environments such as Matlab or Python + Numpy often
encouraging expressing algorithms in terms of vector operations because they
offer a convenient and performant interface. For instance, one can add
2 n-component vectors (a and b) in Numpy with c = a + b.

This is often implemented with something like the following code:

void vec_add(float *left, float *right, float *out, int size) {
  for (int i = 0; i < size; i++) {
    out[i] = left[i] + right[i];
  }
}

Consider the code
a = x + y + z
where x, y, z are n-component vectors.

One way this could be computed would be

vec_add(x, y, a, n);
vec_add(a, z, a, n);

In what ways is this code (2 calls to vec_add) worse than

for (int i = 0; i < n; i++) {
  a[i] = x[i] + y[i] + z[i];
}

? List at least 2 ways (don't need more than a sentence or two for each way).
In the first way, the second call to vec_add depends on the first call to
vec_add so the function calls cannot be paralellized and happen at the same
time. In the second way, the for loop could be unrolled so there are no
instruction depenencies.

The first way has more data IO since we have to read from x and y and write
to a, then read from a and z and write to a so there are 4 reads and 2 writes.
In the second way we read from x, y, and z and write to a so there are 3 reads
and 1 write.