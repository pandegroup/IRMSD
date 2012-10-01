#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <omp.h>
#include <memory.h>
#include "sse_swizzle.h"
#ifdef assert
#undef assert
#define assert(x)
#endif
#ifdef ENABLE_GOTO
extern "C" {
    #include <cblas.h>
}
#endif
static inline double getTimeMilliseconds(void) {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*1000.0 + tv.tv_usec/1000.0;
}

#define DECLARE_METHOD(methodname) \
    void methodname(const int nrealatoms, const int npaddedatoms, const int rowstride, \
                   const float* aT, const float* bT, float* Mout); \
    void block_##methodname(const int nrealatoms, const int npaddedatoms, const int rowstride, \
                         const float* aT, const int refidx, const int Nstrucs, float* Mout);

DECLARE_METHOD(matmul_ihaque)
DECLARE_METHOD(matmul_ihaque_kahan)
DECLARE_METHOD(matmul_ihaque_aos_4w)

void do_bench_with_threads(float* aligned_data, const int N,
                           float* Mstack, const int refidx, const int n_omp_threads);
#ifdef ENABLE_GOTO
void block_matmul_goto(const int nrealatoms, const int npaddedatoms, const int rowstride,
                       float* aT, const int refidx, const int Nstrucs, float* Mout);

void block_matmul_goto_omp(const int nrealatoms, const int npaddedatoms, const int rowstride,
                           float* aT, const int refidx, const int Nstrucs, float* Mout);
#endif

int main(int argc,char** argv) {
    const long N = 3*(1<<26);
    const long totalfloats = (N/3)*4; // 4 to include padding for AoS method
    //+ 16 to guarantee that we can align it to 16 byte boundary
    printf("OMP num procs = %d\n",omp_get_num_procs());
    float* data = (float*)malloc(totalfloats*4LL+16);
    int aligned = ((uintptr_t)data & 0x3F) == 0;
    float* aligned_data = aligned ? data : data + (4 - (((uintptr_t)data)&0xF)/4);
    printf("data at %lx, lower 4 bits are %lx, aligned_data at %lx\n",
           (uintptr_t)data,((uintptr_t)data) & 0x3F,(uintptr_t)aligned_data);
    srandom(1);

    double start = getTimeMilliseconds();
    for (int i = 0; i < totalfloats; i++) {
        data[i] = random()/(float)(RAND_MAX);
    }
    double end = getTimeMilliseconds();
    float sum = 0;
    for (int i = 0; i < totalfloats; i++) {
        sum += data[i];
    }
    printf("Took %.2f ms to compute %ld random numbers = %.2fM/sec\n",
           (float)(end-start),totalfloats,totalfloats/(1000.0f*(end-start)));

    float* Mstack = (float*)malloc(9*1048576*4);

    for (int omp_threads = 1; omp_threads <= 8; omp_threads++) {
        do_bench_with_threads(aligned_data,N,Mstack,0,omp_threads);
    }

    free(data);
    return sum > 0;

}

void do_bench_with_threads(float* aligned_data, const int N,
                           float* Mstack, const int refidx, const int n_omp_threads)
{
    memset(Mstack,0,9*1048576*4);
    omp_set_num_threads(n_omp_threads);
    double start,end;
    int testIH          = 1;
    int testIHKahan     = 0;
    int testGoto        = 0;
    int testGotoOMP     = 0;
    int testIHAoS       = 1;
    int testIHAoSnopad  = 1;
    int testIHAoS4w     = 1;
    int testJDC         = 0;
    int increment = 128;
    const char* output_str = "OMPThreads = %d Took average of %.2f ms to compute " \
                             "%d x %d-atom matrix multiplies with %s method = %.2f GFLOP/s\n";
    double avg_time;
    double gflop_per_sec;
    #define benchmark(method, name) \
        /* One warmup iteration */ \
        method(Natoms,Natoms,Natoms,aligned_data,refidx,Nstrucs,Mstack); \
        start = getTimeMilliseconds(); \
        /* Five real iterations */ \
        method(Natoms,Natoms,Natoms,aligned_data,refidx,Nstrucs,Mstack); \
        method(Natoms,Natoms,Natoms,aligned_data,refidx,Nstrucs,Mstack); \
        method(Natoms,Natoms,Natoms,aligned_data,refidx,Nstrucs,Mstack); \
        method(Natoms,Natoms,Natoms,aligned_data,refidx,Nstrucs,Mstack); \
        method(Natoms,Natoms,Natoms,aligned_data,refidx,Nstrucs,Mstack); \
        end = getTimeMilliseconds(); \
        avg_time = (end - start) / 5.0; \
        gflop_per_sec = FLOPs / (1000000 * avg_time); \
        printf(output_str, n_omp_threads, avg_time, Nstrucs, Natoms, gflop_per_sec, name); \
        fflush(stdout);

    for (int Natoms = 256; Natoms <= 1048576; Natoms += increment) {
        switch (Natoms) {
            case 2048:    increment = 512; break;
            case 8192:    increment = 1024; break;
            case 32768:   increment = 4096; break;
            case 65536:   increment = 16384; break;
            case 262144:  increment = 32768; break;
            case 1048576: increment = 131072; break;
            default: break;
        }
        int Nstrucs = (N/3)/Natoms;
        printf("At %d atoms, we have %d structures\n",Natoms,Nstrucs);
        const long FLOPs = 18L*(long)Nstrucs*(long)Natoms;

        if (testIH) {
            benchmark(block_matmul_ihaque, "axis-major");
        }
        if (testIHKahan) {
            benchmark(block_matmul_ihaque_kahan, "axis-major (Kahan summation)");
        }
        #ifdef ENABLE_GOTO
        if (testGoto) {
            benchmark(block_matmul_goto, "Goto");
        }
        if (testGotoOMP) {
            benchmark(block_matmul_goto_omp, "Goto-OpenMP");
        }
        #endif
        if (testIHAoS4w) {
            benchmark(block_matmul_ihaque_aos_4w, "atom-major");
        }

        printf("\n");
    }

}

static inline void reduction_epilogue(__m128& xx, __m128& xy, __m128& xz,
                                      __m128& yx, __m128& yy, __m128& yz,
                                      __m128& zx, __m128& zy, __m128& zz,
                                      __m128& t0, __m128& t1, __m128& t2)
{
    // Epilogue - reduce 4 wide vectors to one wide
    #ifdef __SSE3__
    // Use SSE3 horizontal add to do the reduction
    /*xmm07 = xx0 xx1 xx2 xx3
      xmm08 = xy0 xy1 xy2 xy3
      xmm09 = xz0 xz1 xz2 xz3
      xmm10 = yx0 yx1 yx2 yx3
      xmm11 = yy0 yy1 yy2 yy3
      xmm12 = yz0 yz1 yz2 yz3
      xmm13 = zx0 zx1 zx2 zx3
      xmm14 = zy0 zy1 zy2 zy3
      xmm15 = zz0 zz1 zz2 zz3
      
      haddps xmm07 xmm08
          xmm07 = xx0+1 xx2+3 xy0+1 xy2+3
      haddps xmm09 xmm10
          xmm09 = xz0+1 xz2+3 yx0+1 yx2+3
      haddps xmm11 xmm12
          xmm11 = yy0+1 yy2+3 yz0+1 yz2+3
      haddps xmm13 xmm14
          xmm13 = zx0+1 zx2+3 zy0+1 zy2+3
      haddps xmm15 xmm14
          xmm15 = zz0+1 zz2+3 zy0+1 zy2+3
      
      haddps xmm07 xmm09
          xmm07 = xx0123 xy0123 xz0123 yx0123
      haddps xmm11 xmm13
          xmm11 = yy0123 yz0123 zx0123 zy0123
      haddps xmm15 xmm09
          xmm15 = zz0123 zy0123 xz0123 yx0123
    */ 
    xx = _mm_hadd_ps(xx,xy);
    xz = _mm_hadd_ps(xz,yx);
    yy = _mm_hadd_ps(yy,yz);
    zx = _mm_hadd_ps(zx,zy);
    zz = _mm_hadd_ps(zz,zy);
    xx = _mm_hadd_ps(xx,xz);
    yy = _mm_hadd_ps(yy,zx);
    
    #else
    // Emulate horizontal adds using SSE2 UNPCKLPS/UNPCKHPS
    t0 = xx;
    t1 = xx;
    t0 = _mm_unpacklo_ps(t0,xz);
        /* = xx0 xz0 xx1 xz1 */
    t1 = _mm_unpackhi_ps(t1,xz);
        /* = xx2 xz2 xx3 xz3 */
    t0 = _mm_add_ps(t0,t1);
        /* = xx02 xz02 xx13 xz13 */
   
    t1 = xy;
    t2 = xy;
    t1 = _mm_unpacklo_ps(t1,yx);
        /* = xy0 yx0 xy1 yx1 */
    t2 = _mm_unpackhi_ps(t2,yx);
        /* = xy2 yx2 xy3 yx3 */
    t1 = _mm_add_ps(t1,t2);
        /* = xy02 yx02 xy13 yx13 */
   
    xx = t0;
    xx = _mm_unpacklo_ps(xx,t1);
        /* = xx02 xy02 xz02 yx02 */
    t0 = _mm_unpackhi_ps(t0,t1);
        /* = xx13 xy13 xz13 yx13 */
    xx = _mm_add_ps(xx,t0);
        /* = xx0123 xy0123 xz0123 yx0123 */
   
    t0 = yy;
    t1 = yy;
    t0 = _mm_unpacklo_ps(t0,zx);
        /* = yy0 zx0 yy1 zx1 */
    t1 = _mm_unpackhi_ps(t1,zx);
        /* = yy2 zx2 yy3 zx3 */
    t0 = _mm_add_ps(t0,t1);
        /* = yy02 zx02 yy13 zx13 */
   
    t1 = yz;
    t2 = yz;
    t1 = _mm_unpacklo_ps(t1,zy);
        /* = yz0 zy0 yz1 zy1 */
    t2 = _mm_unpackhi_ps(t2,zy);
        /* = yz2 zy2 yz3 zy3 */
    t1 = _mm_add_ps(t1,t2);
        /* = yz02 zy02 yz13 zy13 */
   
    yy = t0;
    yy = _mm_unpacklo_ps(yy,t1);
        /* = yy02 yz02 zx02 zy02 */
    t0 = _mm_unpackhi_ps(t0,t1);
        /* = yy13 yz13 zx13 zy13 */
    yy = _mm_add_ps(yy,t0);
        /* = yy0123 yz0123 zx0123 zy0123 */
   
    t1 = _mm_movehl_ps(t1,zz);
        /* = zz2 zz3 - - */
    zz = _mm_add_ps(zz,t1);
        /* = zz02 zz13 - - */
    t1 = _mm_shuffle_ps(zz,zz,_MM_SHUFFLE(1,1,1,1));
        /* = zz13 zz13 zz13 zz13 */
    zz = _mm_add_ps(zz,t1);
        /* = zz0123 zz1133 - - */
    #endif
    return;
}

void matmul_ihaque(const int nrealatoms, const int npaddedatoms, const int rowstride,
                   const float* aT, const float* bT, float* Mout)
{
	int nIndex;
    // Will have 3 garbage elements at the end
    float M[12] __attribute__ ((aligned (16)));

    const float* aTx = aT;
    const float* aTy = aT+rowstride;
    const float* aTz = aT+2*rowstride;
    const float* bTx = bT;
    const float* bTy = bT+rowstride;
    const float* bTz = bT+2*rowstride;


    // npaddedatoms must be a multiple of 4
    int niters = npaddedatoms >> 2;
    __m128 xx,xy,xz,yx,yy,yz,zx,zy,zz;
    __m128 ax,ay,az,b;
    __m128 t0,t1,t2;
    // Prologue
    {
    xx = _mm_setzero_ps();
    xy = _mm_setzero_ps();
    xz = _mm_setzero_ps();
    yx = _mm_setzero_ps();
    yy = _mm_setzero_ps();
    yz = _mm_setzero_ps();
    zx = _mm_setzero_ps();
    zy = _mm_setzero_ps();
    zz = _mm_setzero_ps();
    }

    for (int k = 0; k < niters; k++) {
        ax = _mm_load_ps(aTx);
        ay = _mm_load_ps(aTy);
        az = _mm_load_ps(aTz);

        b = _mm_load_ps(bTx);
        t0 = ax;
        t1 = ay;
        t2 = az;

        t0 = _mm_mul_ps(t0,b);
        t1 = _mm_mul_ps(t1,b);
        t2 = _mm_mul_ps(t2,b);

        xx = _mm_add_ps(xx,t0);
        yx = _mm_add_ps(yx,t1);
        zx = _mm_add_ps(zx,t2);

        b = _mm_load_ps(bTy);
        t0 = ax;
        t1 = ay;
        t2 = az;
        
        t0 = _mm_mul_ps(t0,b);
        t1 = _mm_mul_ps(t1,b);
        t2 = _mm_mul_ps(t2,b);

        xy = _mm_add_ps(xy,t0);
        yy = _mm_add_ps(yy,t1);
        zy = _mm_add_ps(zy,t2);

        b = _mm_load_ps(bTz);

        ax = _mm_mul_ps(ax,b);
        ay = _mm_mul_ps(ay,b);
        az = _mm_mul_ps(az,b);
        
        xz = _mm_add_ps(xz,ax);
        yz = _mm_add_ps(yz,ay);
        zz = _mm_add_ps(zz,az);

        aTx += 4;
        aTy += 4;
        aTz += 4;
        bTx += 4;
        bTy += 4;
        bTz += 4;
    }
    reduction_epilogue(xx, xy, xz, yx, yy, yz, zx, zy, zz, t0, t1, t2);

    _mm_storeu_ps(Mout  , xx);
    _mm_storeu_ps(Mout+4, yy);
    _mm_store_ss( Mout+8, zz);

    return;
}

void block_matmul_ihaque(const int nrealatoms, const int npaddedatoms, const int rowstride,
                         const float* aT, const int refidx, const int Nstrucs, float* Mout)
{
    #pragma omp parallel for
    for (int i = 0; i < Nstrucs; i++) {
        matmul_ihaque(nrealatoms,npaddedatoms,rowstride,
                      aT+3*refidx*rowstride,aT+3*i*rowstride,Mout+9*i);
    }

    return;
}

void matmul_ihaque_kahan(const int nrealatoms, const int npaddedatoms, const int rowstride,
                         const float* aT, const float* bT, float* Mout)
{
    /* Modification of matmul_ihaque SoA method to use Kahan compensated summation */
	int nIndex;
    // Will have 3 garbage elements at the end
    float M[12] __attribute__ ((aligned (16)));

    const float* aTx = aT;
    const float* aTy = aT+rowstride;
    const float* aTz = aT+2*rowstride;
    const float* bTx = bT;
    const float* bTy = bT+rowstride;
    const float* bTz = bT+2*rowstride;


    // npaddedatoms must be a multiple of 4
    int niters = npaddedatoms >> 2;
    __m128 xx,xy,xz,yx,yy,yz,zx,zy,zz;
    __m128 cxx,cxy,cxz,cyx,cyy,cyz,czx,czy,czz;
    __m128 ax,ay,az,b;
    __m128 t0,t1,t2,t3,t4,t5;
    // Prologue
    {
    xx = _mm_setzero_ps();
    xy = _mm_setzero_ps();
    cxx = _mm_setzero_ps();
    cxy = _mm_setzero_ps();
    xz = _mm_setzero_ps();
    yx = _mm_setzero_ps();
    cxz = _mm_setzero_ps();
    cyx = _mm_setzero_ps();
    yy = _mm_setzero_ps();
    cyy = _mm_setzero_ps();
    yz = _mm_setzero_ps();
    zx = _mm_setzero_ps();
    cyz = _mm_setzero_ps();
    czx = _mm_setzero_ps();
    zy = _mm_setzero_ps();
    zz = _mm_setzero_ps();
    czy = _mm_setzero_ps();
    czz = _mm_setzero_ps();
    }

    for (int k = 0; k < niters; k++) {
        ax = _mm_load_ps(aTx);
        ay = _mm_load_ps(aTy);
        az = _mm_load_ps(aTz);

        b = _mm_load_ps(bTx);
        t0 = ax;
        t1 = ay;
        t2 = az;

        t0 = _mm_mul_ps(t0,b);
        t1 = _mm_mul_ps(t1,b);
        t2 = _mm_mul_ps(t2,b);
        t0 = _mm_sub_ps(t0,cxx);
        t1 = _mm_sub_ps(t1,cyx);
        t2 = _mm_sub_ps(t2,czx);
        t3 = xx;
        t4 = yx;
        t5 = zx;
        t3 = _mm_add_ps(t3,t0);
        t4 = _mm_add_ps(t4,t1);
        t5 = _mm_add_ps(t5,t2);
        cxx = t3;
        cyx = t4;
        czx = t5;
        cxx = _mm_sub_ps(cxx,xx);
        cyx = _mm_sub_ps(cyx,yx);
        czx = _mm_sub_ps(czx,zx);
        cxx = _mm_sub_ps(cxx,t0);
        cyx = _mm_sub_ps(cyx,t1);
        czx = _mm_sub_ps(czx,t2);
        xx = t3;
        yx = t4;
        zx = t5;

        b = _mm_load_ps(bTy);
        t0 = ax;
        t1 = ay;
        t2 = az;
        
        t0 = _mm_mul_ps(t0,b);
        t1 = _mm_mul_ps(t1,b);
        t2 = _mm_mul_ps(t2,b);
        t0 = _mm_sub_ps(t0,cxy);
        t1 = _mm_sub_ps(t1,cyy);
        t2 = _mm_sub_ps(t2,czy);
        t3 = xy;
        t4 = yy;
        t5 = zy;
        t3 = _mm_add_ps(t3,t0);
        t4 = _mm_add_ps(t4,t1);
        t5 = _mm_add_ps(t5,t2);
        cxy = t3;
        cyy = t4;
        czy = t5;
        cxy = _mm_sub_ps(cxy,xy);
        cyy = _mm_sub_ps(cyy,yy);
        czy = _mm_sub_ps(czy,zy);
        cxy = _mm_sub_ps(cxy,t0);
        cyy = _mm_sub_ps(cyy,t1);
        czy = _mm_sub_ps(czy,t2);
        xy = t3;
        yy = t4;
        zy = t5;

        b = _mm_load_ps(bTz);
        t0 = ax;
        t1 = ay;
        t2 = az;

        t0 = _mm_mul_ps(t0,b);
        t1 = _mm_mul_ps(t1,b);
        t2 = _mm_mul_ps(t2,b);
        t0 = _mm_sub_ps(t0,cxz);
        t1 = _mm_sub_ps(t1,cyz);
        t2 = _mm_sub_ps(t2,czz);
        t3 = xz;
        t4 = yz;
        t5 = zz;
        t3 = _mm_add_ps(t3,t0);
        t4 = _mm_add_ps(t4,t1);
        t5 = _mm_add_ps(t5,t2);
        cxz = t3;
        cyz = t4;
        czz = t5;
        cxz = _mm_sub_ps(cxz,xz);
        cyz = _mm_sub_ps(cyz,yz);
        czz = _mm_sub_ps(czz,zz);
        cxz = _mm_sub_ps(cxz,t0);
        cyz = _mm_sub_ps(cyz,t1);
        czz = _mm_sub_ps(czz,t2);
        xz = t3;
        yz = t4;
        zz = t5;
        
        aTx += 4;
        aTy += 4;
        aTz += 4;
        bTx += 4;
        bTy += 4;
        bTz += 4;
    }
    reduction_epilogue(xx, xy, xz, yx, yy, yz, zx, zy, zz, t0, t1, t2);

    _mm_storeu_ps(Mout  , xx);
    _mm_storeu_ps(Mout+4, yy);
    _mm_store_ss( Mout+8, zz);

    return;
}

void block_matmul_ihaque_kahan(const int nrealatoms, const int npaddedatoms, const int rowstride,
                               const float* aT, const int refidx, const int Nstrucs, float* Mout)
{
    #pragma omp parallel for
    for (int i = 0; i < Nstrucs; i++) {
        matmul_ihaque_kahan(nrealatoms,npaddedatoms,rowstride,
                            aT+3*refidx*rowstride,aT+3*i*rowstride,Mout+9*i);
    }

    return;
}

#ifdef ENABLE_GOTO
void block_matmul_goto(const int nrealatoms, const int npaddedatoms, const int rowstride,
                       float* aT, const int refidx, const int Nstrucs, float* Mout)
{
    const int numConfs = Nstrucs;
    const int numAtoms = nrealatoms;
    float* x_nk = aT;
    float* y_nk = aT+3*refidx*rowstride;
    float* MStack = Mout;

    //void cblas_sgemm(const  CBLAS_ORDER Order, const  CBLAS_TRANSPOSE TransA, const  CBLAS_TRANSPOSE TransB,
    //                 const MKL_INT M, const MKL_INT N, const MKL_INT K,
    //                 const float alpha, const float *A, const MKL_INT lda,
    //                 const float *B, const MKL_INT ldb,
    //                 const float beta, float *C, const MKL_INT ldc);
    // A' is M x K
    // B is  K x N
    // C is  M x N
    //printf("Calling cblas_sgemm with M = %d, N = %d, K = %d\n",3,3,numAtoms);
	
	//	int MM=3*numConfs;
	int NN=3;
	//	int KK=numAtoms;
	int MM=3*numConfs;
	int KK=numAtoms;

	float TMP[12]={0.,0.,0.,  0.,0.,0.,  0.,0.,0.,  0.,0.,0.,};

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		    MM,    NN,    KK,
		    1.0f, x_nk, KK,
		    y_nk, KK,
		    0.0f, MStack, NN);
	    
    return;

}

void block_matmul_goto_omp(const int nrealatoms, const int npaddedatoms, const int rowstride,
                           float* aT, const int refidx, const int Nstrucs, float* Mout)
{
    const int numConfs = Nstrucs;
    const int numAtoms = nrealatoms;
    float* x_nk = aT;
    float* y_nk = aT+3*refidx*rowstride;
    float* MStack = Mout;

    //void cblas_sgemm(const  CBLAS_ORDER Order, const  CBLAS_TRANSPOSE TransA, const  CBLAS_TRANSPOSE TransB,
    //                 const MKL_INT M, const MKL_INT N, const MKL_INT K,
    //                 const float alpha, const float *A, const MKL_INT lda,
    //                 const float *B, const MKL_INT ldb,
    //                 const float beta, float *C, const MKL_INT ldc);
    // A' is M x K
    // B is  K x N
    // C is  M x N
    //printf("Calling cblas_sgemm with M = %d, N = %d, K = %d\n",3,3,numAtoms);
	
	//	int MM=3*numConfs;
	int NN=3;
	//	int KK=numAtoms;
	int MM=3;
	int KK=numAtoms;

    #pragma omp parallel
    {
        int threadIdx = omp_get_thread_num();
        int nThreads  = omp_get_num_threads();
        int strucPerThread = numConfs/nThreads;
        int threadBase  = strucPerThread*threadIdx;
        int threadBound = (threadIdx == nThreads-1) ? numConfs : threadBase + strucPerThread;
        int myNrows = (threadBound-threadBase)*3;
        //#pragma omp critical
        //printf("In OMP thread %d, working on confs %d - %d = %d rows\n",threadIdx,threadBase,threadBound,myNrows);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            myNrows,    NN,    KK,
   	        1.0f, x_nk+3*threadBase*rowstride, KK,
            y_nk, KK,
	        0.0f, MStack+9*threadBase, NN);
    }
    return;
}
#endif

static inline void aos_deinterleaved_load(const float* S, __m128& x, __m128& y, __m128& z)
{
    __m128 t1, t2;
    x  = _mm_load_ps(S);
    y  = _mm_load_ps(S+4);
    t1 = _mm_load_ps(S+8);
    z  = x;
    t2 = y;
    
    t2 = _mm_shuffle_ps_yzyz(t2,t1);
    z  = _mm_shuffle_ps_yzxw(z,t2);
    x  = _mm_shuffle_ps_xwyz(x,t2);
    y  = _mm_shuffle_ps_xwxw(y,z);
    z  = _mm_shuffle_ps_yzxw(z,t1);
    y  = _mm_swizzle_ps_zxyw(y);
    return;
}

void matmul_ihaque_aos_4w(const int nrealatoms, const int npaddedatoms, const int rowstride,
                          const float* a, const float* b, float* Mout)
{
    // Will have 3 garbage elements at the end
    float M[12] __attribute__ ((aligned (16)));

    assert(nrealatoms % 4 == 0);

    // npaddedatoms must be a multiple of 4
    int niters = nrealatoms >> 2;
    __m128 xx,xy,xz,yx,yy,yz,zx,zy,zz;
    __m128 ax,ay,az,bx,by,bz;
    __m128 t0,t1,t2,t3;
    // Prologue
    {
    xx = _mm_setzero_ps();
    xy = _mm_setzero_ps();
    xz = _mm_setzero_ps();
    yx = _mm_setzero_ps();
    yy = _mm_setzero_ps();
    yz = _mm_setzero_ps();
    zx = _mm_setzero_ps();
    zy = _mm_setzero_ps();
    zz = _mm_setzero_ps();
    }
    for (int k = 0; k < niters; k++)
    {
        aos_deinterleaved_load(b,bx,by,bz);
        aos_deinterleaved_load(a,ax,ay,az);

        t1 = bx;
        t2 = by;
        t3 = bz;
        t1 = _mm_mul_ps(t1,ax);
        t2 = _mm_mul_ps(t2,ax);
        t3 = _mm_mul_ps(t3,ax);
        xx = _mm_add_ps(xx,t1);
        xy = _mm_add_ps(xy,t2);
        xz = _mm_add_ps(xz,t3);

        t1 = bx;
        t2 = by;
        t3 = bz;
        t1 = _mm_mul_ps(t1,ay);
        t2 = _mm_mul_ps(t2,ay);
        t3 = _mm_mul_ps(t3,ay);
        yx = _mm_add_ps(yx,t1);
        yy = _mm_add_ps(yy,t2);
        yz = _mm_add_ps(yz,t3);

        bx = _mm_mul_ps(bx,az);
        by = _mm_mul_ps(by,az);
        bz = _mm_mul_ps(bz,az);
        zx = _mm_add_ps(zx,bx);
        zy = _mm_add_ps(zy,by);
        zz = _mm_add_ps(zz,bz);

        a += 12;
        b += 12;
    }
    reduction_epilogue(xx, xy, xz, yx, yy, yz, zx, zy, zz, t0, t1, t2);

    _mm_storeu_ps(Mout  , xx);
    _mm_storeu_ps(Mout+4, yy);
    _mm_store_ss( Mout+8, zz);

    return;
}

void block_matmul_ihaque_aos_4w(const int nrealatoms, const int npaddedatoms, const int rowstride,
                                   const float* aT, const int refidx, const int Nstrucs, float* Mout)
{
    #pragma omp parallel for
    for (int i = 0; i < Nstrucs; i++) {
        matmul_ihaque_aos_4w(nrealatoms,npaddedatoms,rowstride,
                             aT+3*refidx*nrealatoms,aT+3*i*nrealatoms,Mout+9*i);
    }

    return;
}
