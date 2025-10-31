/********************************************************************
 *  image_pthread.c
 *
 *  Parallel version of the image convolution program using pthreads.
 *  The convolution is split across several threads, each processing
 *  a contiguous block of image rows.
 *
 *  Compile with:
 *      gcc -g image_pthread.c -o image_pthread -lm -lpthread
 ********************************************************************/

#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <pthread.h>          /* <-- NEW: pthreads header */
#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/* -----------------------------------------------------------------
 *  Kernel matrices (same as the original program)
 * ----------------------------------------------------------------- */
Matrix algorithms[] = {
    {{0,-1,0},{-1,4,-1},{0,-1,0}},                     /* EDGE      */
    {{0,-1,0},{-1,5,-1},{0,-1,0}},                     /* SHARPEN   */
    {{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0}}, /* BLUR      */
    {{1.0/16,1.0/8,1.0/16},{1.0/8,1.0/4,1.0/8},{1.0/16,1.0/8,1.0/16}}, /* GAUSS     */
    {{-2,-1,0},{-1,1,1},{0,1,2}},                       /* EMBOSS    */
    {{0,0,0},{0,1,0},{0,0,0}}                          /* IDENTITY  */
};

/* -----------------------------------------------------------------
 *  getPixelValue - unchanged from original
 * ----------------------------------------------------------------- */
uint8_t getPixelValue(Image* srcImage, int x, int y, int bit, Matrix algorithm)
{
    int px = x+1, py = y+1, mx = x-1, my = y-1;

    if (mx < 0) mx = 0;
    if (my < 0) my = 0;
    if (px >= srcImage->width)  px = srcImage->width-1;
    if (py >= srcImage->height) py = srcImage->height-1;

    uint8_t result =
        algorithm[0][0]*srcImage->data[Index(mx,my,srcImage->width,bit,srcImage->bpp)] +
        algorithm[0][1]*srcImage->data[Index(x ,my,srcImage->width,bit,srcImage->bpp)] +
        algorithm[0][2]*srcImage->data[Index(px,my,srcImage->width,bit,srcImage->bpp)] +
        algorithm[1][0]*srcImage->data[Index(mx,y ,srcImage->width,bit,srcImage->bpp)] +
        algorithm[1][1]*srcImage->data[Index(x ,y ,srcImage->width,bit,srcImage->bpp)] +
        algorithm[1][2]*srcImage->data[Index(px,y ,srcImage->width,bit,srcImage->bpp)] +
        algorithm[2][0]*srcImage->data[Index(mx,py,srcImage->width,bit,srcImage->bpp)] +
        algorithm[2][1]*srcImage->data[Index(x ,py,srcImage->width,bit,srcImage->bpp)] +
        algorithm[2][2]*srcImage->data[Index(px,py,srcImage->width,bit,srcImage->bpp)];

    return result;
}

/* -----------------------------------------------------------------
 *  Thread argument structure
 * ----------------------------------------------------------------- */
typedef struct {
    Image*  srcImage;
    Image*  destImage;
    Matrix  algorithm;
    int     start_row;
    int     end_row;
} ThreadArg;

/* -----------------------------------------------------------------
 *  Thread routine - convolves a block of rows
 * ----------------------------------------------------------------- */
void* thread_convolute(void* arg)
{
    ThreadArg* ta = (ThreadArg*)arg;
    int pix, bit;

    for (int row = ta->start_row; row < ta->end_row; ++row) {
        for (pix = 0; pix < ta->srcImage->width; ++pix) {
            for (bit = 0; bit < ta->srcImage->bpp; ++bit) {
                ta->destImage->data[Index(pix, row,
                                          ta->srcImage->width,
                                          bit, ta->srcImage->bpp)] =
                    getPixelValue(ta->srcImage, pix, row, bit, ta->algorithm);
            }
        }
    }
    return NULL;
}

/* -----------------------------------------------------------------
 *  convolute - now creates and joins pthreads
 * ----------------------------------------------------------------- */
void convolute(Image* srcImage, Image* destImage, Matrix algorithm)
{
    const int num_threads = 4;               /* <-- change if you want more/less */
    pthread_t threads[num_threads];
    ThreadArg args[num_threads];

    int rows_per_thread = srcImage->height / num_threads;
    int remainder       = srcImage->height % num_threads;
    int current_start   = 0;

    for (int i = 0; i < num_threads; ++i) {
        args[i].srcImage   = srcImage;
        args[i].destImage  = destImage;
        memcpy(args[i].algorithm, algorithm, sizeof(Matrix));

        args[i].start_row = current_start;
        int extra = (i < remainder) ? 1 : 0;
        args[i].end_row   = current_start + rows_per_thread + extra;
        current_start     = args[i].end_row;

        pthread_create(&threads[i], NULL, thread_convolute, &args[i]);
    }

    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }
}

/* -----------------------------------------------------------------
 *  Usage & GetKernelType - unchanged
 * ----------------------------------------------------------------- */
int Usage(void)
{
    printf("Usage: image <filename> <type>\n"
           "\twhere type is one of (edge,sharpen,blur,gauss,emboss,identity)\n");
    return -1;
}

enum KernelTypes GetKernelType(char* type)
{
    if (!strcmp(type,"edge"))    return EDGE;
    if (!strcmp(type,"sharpen")) return SHARPEN;
    if (!strcmp(type,"blur"))    return BLUR;
    if (!strcmp(type,"gauss"))   return GAUSE_BLUR;
    if (!strcmp(type,"emboss"))  return EMBOSS;
    return IDENTITY;
}

/* -----------------------------------------------------------------
 *  main
 * ----------------------------------------------------------------- */
int main(int argc, char** argv)
{
    long t1 = time(NULL);

    stbi_set_flip_vertically_on_load(0);
    if (argc != 3) return Usage();

    char* fileName = argv[1];
    if (!strcmp(argv[1],"pic4.jpg") && !strcmp(argv[2],"gauss")) {
        printf("You have applied a gaussian filter to Gauss which has caused a tear in the time-space continum.\n");
    }

    enum KernelTypes type = GetKernelType(argv[2]);

    Image srcImage, destImage;

    srcImage.data = stbi_load(fileName,
                              &srcImage.width,
                              &srcImage.height,
                              &srcImage.bpp, 0);
    if (!srcImage.data) {
        printf("Error loading file %s.\n", fileName);
        return -1;
    }

    destImage.bpp    = srcImage.bpp;
    destImage.height = srcImage.height;
    destImage.width  = srcImage.width;
    destImage.data   = malloc(sizeof(uint8_t) *
                              destImage.width * destImage.bpp * destImage.height);

    /* ---- parallel convolution ---- */
    convolute(&srcImage, &destImage, algorithms[type]);

    stbi_write_png("output.png",
                   destImage.width, destImage.height,
                   destImage.bpp, destImage.data,
                   destImage.bpp * destImage.width);

    stbi_image_free(srcImage.data);
    free(destImage.data);

    long t2 = time(NULL);
    printf("Took %ld seconds\n", t2 - t1);
    return 0;
}
