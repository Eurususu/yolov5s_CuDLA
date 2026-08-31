 /*
  * Header file for NvSciBuf APIs
  *
  * Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
  *
  * NVIDIA Corporation and its licensors retain all intellectual property
  * and proprietary rights in and to this software, related documentation
  * and any modifications thereto.  Any use, reproduction, disclosure or
  * distribution of this software and related documentation without an express
  * license agreement from NVIDIA Corporation is strictly prohibited.
  */
 #ifndef INCLUDED_NVSCIBUF_H
 #define INCLUDED_NVSCIBUF_H

 #include <stddef.h>
 #include <stdbool.h>
 #include <stdint.h>
 #include "nvscierror.h"
 #include "nvsciipc.h"

 #if defined(__cplusplus)
 extern "C"
 {
 #endif

 #if defined __GNUC__
     #define PACK_BUF( __Declaration__ ) __Declaration__ __attribute__((packed))
 #else
     #define PACK_BUF( __Declaration__ ) __pragma(pack(push, 1)) __Declaration__ __pragma(pack(pop))
 #endif

 typedef enum {
     NvSciBufType_General = 0U,
     NvSciBufType_RawBuffer = 1U,
     NvSciBufType_Image = 2U,
     NvSciBufType_Tensor = 3U,
     NvSciBufType_Array = 4U,
     NvSciBufType_Pyramid = 5U,
     NvSciBufType_MaxValid = 6U,
     NvSciBufType_UpperBound = 6U,
 } NvSciBufType;

 static const uint32_t NvSciBufMajorVersion = 2U;

 static const uint32_t NvSciBufMinorVersion = 11U;

 #if defined(__cplusplus)

 static const int NV_SCI_BUF_TENSOR_MAX_DIMS = 8;

 static const int NV_SCI_BUF_IMAGE_MAX_PLANES = 3;

 static const int NV_SCI_BUF_PYRAMID_MAX_LEVELS = 10;

 static const int NVSCIBUF_EXPORT_DESC_SIZE = 32;

 static const uint32_t NV_SCI_BUF_PEER_INFO_MAX_NUMBER = 128U;

 static const uint32_t NV_SCI_BUF_PEER_INFO_SELF_SOCID = 0xFFFFFFFFu;

 static const uint32_t NV_SCI_BUF_PEER_INFO_SELF_VMID = 0xFFFFFFFFu;

 static const int NV_SCI_BUF_PEER_HW_ENGINE_MAX_NUMBER = 128U;

 static const uint32_t NV_SCI_BUF_MAX_GPUS = 16u;

 static const int NV_SCI_BUF_ATTRKEY_BIT_COUNT = 16;

 static const int NV_SCI_BUF_DATATYPE_BIT_COUNT = 10;

 static const int NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC = 0;

 /*
  * @brief Global constant to specify the start-bit of attribute Keytype.
  */
 static const int NV_SCI_BUF_KEYTYPE_BIT_START =
         (NV_SCI_BUF_DATATYPE_BIT_COUNT + NV_SCI_BUF_ATTRKEY_BIT_COUNT);

 static const int NV_SCI_BUF_GENERAL_ATTR_KEY_START =
            (NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << NV_SCI_BUF_KEYTYPE_BIT_START) |
            (NvSciBufType_General << NV_SCI_BUF_ATTRKEY_BIT_COUNT);

 static const int NV_SCI_BUF_RAW_BUF_ATTR_KEY_START =
            (NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << NV_SCI_BUF_KEYTYPE_BIT_START) |
            (NvSciBufType_RawBuffer << NV_SCI_BUF_ATTRKEY_BIT_COUNT);

 static const int NV_SCI_BUF_IMAGE_ATTR_KEY_START =
            (NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << NV_SCI_BUF_KEYTYPE_BIT_START) |
            (NvSciBufType_Image << NV_SCI_BUF_ATTRKEY_BIT_COUNT);

 static const int NV_SCI_BUF_PYRAMID_ATTR_KEY_START =
            (NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << NV_SCI_BUF_KEYTYPE_BIT_START) |
            (NvSciBufType_Pyramid << NV_SCI_BUF_ATTRKEY_BIT_COUNT);

 static const int NV_SCI_BUF_ARRAY_ATTR_KEY_START =
            (NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << NV_SCI_BUF_KEYTYPE_BIT_START) |
            (NvSciBufType_Array << NV_SCI_BUF_ATTRKEY_BIT_COUNT);

 static const int NV_SCI_BUF_TENSOR_ATTR_KEY_START =
            (NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << NV_SCI_BUF_KEYTYPE_BIT_START) |
            (NvSciBufType_Tensor << NV_SCI_BUF_ATTRKEY_BIT_COUNT);

 #else

 #define NV_SCI_BUF_TENSOR_MAX_DIMS  8u

 #define NV_SCI_BUF_IMAGE_MAX_PLANES 3u

 #define NV_SCI_BUF_PYRAMID_MAX_LEVELS 10u

 #define NVSCIBUF_EXPORT_DESC_SIZE   32u

 #define NV_SCI_BUF_PEER_INFO_MAX_NUMBER 128u

 #define NV_SCI_BUF_PEER_INFO_SELF_SOCID 0xFFFFFFFFu

 #define NV_SCI_BUF_PEER_INFO_SELF_VMID 0xFFFFFFFFu

 #define NV_SCI_BUF_PEER_HW_ENGINE_MAX_NUMBER  128U

 #define NV_SCI_BUF_MAX_GPUS 16

 #define NV_SCI_BUF_ATTRKEY_BIT_COUNT  16u

 #define NV_SCI_BUF_DATATYPE_BIT_COUNT  10u

 #define NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC 0u

 #define NV_SCI_BUF_KEYTYPE_BIT_START \
         (NV_SCI_BUF_DATATYPE_BIT_COUNT + NV_SCI_BUF_ATTRKEY_BIT_COUNT)

 #define NV_SCI_BUF_GENERAL_ATTR_KEY_START \
         ((NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << NV_SCI_BUF_KEYTYPE_BIT_START) | \
         (NvSciBufType_General << NV_SCI_BUF_ATTRKEY_BIT_COUNT))

 #define NV_SCI_BUF_RAW_BUF_ATTR_KEY_START \
           ((NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << NV_SCI_BUF_KEYTYPE_BIT_START) | \
           (NvSciBufType_RawBuffer << NV_SCI_BUF_ATTRKEY_BIT_COUNT))

 #define NV_SCI_BUF_IMAGE_ATTR_KEY_START \
           ((NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << NV_SCI_BUF_KEYTYPE_BIT_START) | \
           (NvSciBufType_Image << NV_SCI_BUF_ATTRKEY_BIT_COUNT))

 #define NV_SCI_BUF_PYRAMID_ATTR_KEY_START \
           ((NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << NV_SCI_BUF_KEYTYPE_BIT_START) | \
           (NvSciBufType_Pyramid << NV_SCI_BUF_ATTRKEY_BIT_COUNT))

 #define NV_SCI_BUF_ARRAY_ATTR_KEY_START \
           ((NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << NV_SCI_BUF_KEYTYPE_BIT_START) | \
           (NvSciBufType_Array << NV_SCI_BUF_ATTRKEY_BIT_COUNT))

 #define NV_SCI_BUF_TENSOR_ATTR_KEY_START \
           ((NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << NV_SCI_BUF_KEYTYPE_BIT_START) | \
           (NvSciBufType_Tensor << NV_SCI_BUF_ATTRKEY_BIT_COUNT))

 #endif

 typedef enum {
     NvSciBufAttrKey_LowerBound =         NV_SCI_BUF_GENERAL_ATTR_KEY_START,

     NvSciBufGeneralAttrKey_Types,

     NvSciBufGeneralAttrKey_NeedCpuAccess,

     NvSciBufGeneralAttrKey_RequiredPerm,

     NvSciBufGeneralAttrKey_EnableCpuCache,

     NvSciBufGeneralAttrKey_GpuId,

     NvSciBufGeneralAttrKey_CpuNeedSwCacheCoherency,

     NvSciBufGeneralAttrKey_ActualPerm,

     NvSciBufGeneralAttrKey_VidMem_GpuId,

     NvSciBufGeneralAttrKey_EnableGpuCache,

     NvSciBufGeneralAttrKey_GpuSwNeedCacheCoherency,

     NvSciBufGeneralAttrKey_EnableGpuCompression,

     NvSciBufGeneralAttrKey_PeerLocationInfo,

     NvSciBufGeneralAttrKey_PeerHwEngineArray,

     NvSciBufRawBufferAttrKey_Size   =  NV_SCI_BUF_RAW_BUF_ATTR_KEY_START,

     NvSciBufRawBufferAttrKey_Align,

     NvSciBufImageAttrKey_Layout   =    NV_SCI_BUF_IMAGE_ATTR_KEY_START,

     NvSciBufImageAttrKey_TopPadding,

     NvSciBufImageAttrKey_BottomPadding,

     NvSciBufImageAttrKey_LeftPadding,

     NvSciBufImageAttrKey_RightPadding,

     NvSciBufImageAttrKey_VprFlag,

     NvSciBufImageAttrKey_Size,

     NvSciBufImageAttrKey_Alignment,

     NvSciBufImageAttrKey_PlaneCount,

     NvSciBufImageAttrKey_PlaneColorFormat,

     NvSciBufImageAttrKey_PlaneColorStd,

     NvSciBufImageAttrKey_PlaneBaseAddrAlign,

     NvSciBufImageAttrKey_PlaneWidth,

     NvSciBufImageAttrKey_PlaneHeight,

     NvSciBufImageAttrKey_PlaneScanType = 0x2000e,
     NvSciBufImageAttrKey_ScanType = NvSciBufImageAttrKey_PlaneScanType,

     NvSciBufImageAttrKey_PlaneBitsPerPixel,

     NvSciBufImageAttrKey_PlaneOffset,

     NvSciBufImageAttrKey_PlaneDatatype,

     NvSciBufImageAttrKey_PlaneChannelCount,

     NvSciBufImageAttrKey_PlaneSecondFieldOffset,

     NvSciBufImageAttrKey_PlanePitch,

     NvSciBufImageAttrKey_PlaneAlignedHeight,

     NvSciBufImageAttrKey_PlaneAlignedSize,

     NvSciBufImageAttrKey_ImageCount,

     NvSciBufImageAttrKey_SurfType,

     NvSciBufImageAttrKey_SurfMemLayout,

     NvSciBufImageAttrKey_SurfSampleType,

     NvSciBufImageAttrKey_SurfBPC,

     NvSciBufImageAttrKey_SurfComponentOrder,

     NvSciBufImageAttrKey_SurfWidthBase,

     NvSciBufImageAttrKey_SurfHeightBase,

     NvSciBufImageAttrKey_SurfColorStd,

     NvSciBufTensorAttrKey_DataType  =  NV_SCI_BUF_TENSOR_ATTR_KEY_START,

     NvSciBufTensorAttrKey_NumDims,

     NvSciBufTensorAttrKey_SizePerDim,

     NvSciBufTensorAttrKey_AlignmentPerDim,

     NvSciBufTensorAttrKey_StridesPerDim,

     NvSciBufTensorAttrKey_PixelFormat,

     NvSciBufTensorAttrKey_BaseAddrAlign,

     NvSciBufTensorAttrKey_Size,

     NvSciBufArrayAttrKey_DataType   =  NV_SCI_BUF_ARRAY_ATTR_KEY_START,

     NvSciBufArrayAttrKey_Stride,

     NvSciBufArrayAttrKey_Capacity,

     NvSciBufArrayAttrKey_Size,

     NvSciBufArrayAttrKey_Alignment,

     NvSciBufPyramidAttrKey_NumLevels  =  NV_SCI_BUF_PYRAMID_ATTR_KEY_START,

     NvSciBufPyramidAttrKey_Scale,

     NvSciBufPyramidAttrKey_LevelOffset,

     NvSciBufPyramidAttrKey_LevelSize,

     NvSciBufPyramidAttrKey_Alignment,

     NvSciBufAttrKey_UpperBound = 0x3ffffffU,

 } NvSciBufAttrKey;

 typedef enum {
     NvSciBufAccessPerm_Readonly = 1,
     NvSciBufAccessPerm_ReadWrite = 3,
     NvSciBufAccessPerm_Auto,
     NvSciBufAccessPerm_Invalid,
 } NvSciBufAttrValAccessPerm;

 typedef enum {
     NvSciBufImage_BlockLinearType,
     NvSciBufImage_PitchLinearType,
 } NvSciBufAttrValImageLayoutType;

 typedef enum {
     NvSciBufScan_ProgressiveType,
     NvSciBufScan_InterlaceType,
 } NvSciBufAttrValImageScanType;

 typedef enum {
     NvSciColor_LowerBound,
     /* RAW PACKED */
     /* Bit ordering for little endian machine is as follows
      * for NvSciColor_X12Bayer20GBRG
      * pattern  BBBBBBBB BBBBBBBB BBBB**** ******** GGGGGGGG GGGGGGGG GGGG**** ********
      *          +------+ +------+ +------+ +------+ +------+ +------+ +------+ +------+
      * bit      63    56 55    48 47    40 39    32 31    24 23    16 15     8 7      0
      *          +---------------------------------+ +---------------------------------+
      * pixel                     1                                   0
      *          +---------------------------------------------------------------------+
      * pitch                                       0
      *
      * pattern  GGGGGGGG GGGGGGGG GGGG**** ******** RRRRRRRR RRRRRRRR RRRR**** ********
      *          +------+ +------+ +------+ +------+ +------+ +------+ +------+ +------+
      * bit      63    56 55    48 47    40 39    32 31    24 23    16 15     8 7      0
      *          +---------------------------------+ +---------------------------------+
      * pixel                     1                                   0
      *          +---------------------------------------------------------------------+
      * pitch                                       1
      *
      * '*' means undefined bit value
      */
     NvSciColor_Bayer8RGGB,
     NvSciColor_Bayer8CCCC,
     NvSciColor_Bayer8BGGR,
     NvSciColor_Bayer8GBRG,
     NvSciColor_Bayer8GRBG,
     NvSciColor_Bayer16BGGR,
     NvSciColor_Bayer16CCCC,
     NvSciColor_Bayer16GBRG,
     NvSciColor_Bayer16GRBG,
     NvSciColor_Bayer16RGGB,
     NvSciColor_Bayer16RCCB,
     NvSciColor_Bayer16BCCR,
     NvSciColor_Bayer16CRBC,
     NvSciColor_Bayer16CBRC,
     NvSciColor_Bayer16RCCC,
     NvSciColor_Bayer16CCCR,
     NvSciColor_Bayer16CRCC,
     NvSciColor_Bayer16CCRC,
     NvSciColor_X2Bayer14GBRG,
     NvSciColor_X4Bayer12GBRG,
     NvSciColor_X6Bayer10GBRG,
     NvSciColor_X2Bayer14GRBG,
     NvSciColor_X4Bayer12GRBG,
     NvSciColor_X6Bayer10GRBG,
     NvSciColor_X2Bayer14BGGR,
     NvSciColor_X4Bayer12BGGR,
     NvSciColor_X6Bayer10BGGR,
     NvSciColor_X2Bayer14RGGB,
     NvSciColor_X4Bayer12RGGB,
     NvSciColor_X6Bayer10RGGB,
     NvSciColor_X2Bayer14CCCC,
     NvSciColor_X4Bayer12CCCC,
     NvSciColor_X6Bayer10CCCC,
     NvSciColor_X4Bayer12RCCB,
     NvSciColor_X4Bayer12BCCR,
     NvSciColor_X4Bayer12CRBC,
     NvSciColor_X4Bayer12CBRC,
     NvSciColor_X4Bayer12RCCC,
     NvSciColor_X4Bayer12CCCR,
     NvSciColor_X4Bayer12CRCC,
     NvSciColor_X4Bayer12CCRC,
     NvSciColor_Signed_X2Bayer14CCCC,
     NvSciColor_Signed_X4Bayer12CCCC,
     NvSciColor_Signed_X6Bayer10CCCC,
     NvSciColor_Signed_Bayer16CCCC,
     NvSciColor_FloatISP_Bayer16CCCC,
     NvSciColor_FloatISP_Bayer16RGGB,
     NvSciColor_FloatISP_Bayer16BGGR,
     NvSciColor_FloatISP_Bayer16GRBG,
     NvSciColor_FloatISP_Bayer16GBRG,
     NvSciColor_FloatISP_Bayer16RCCB,
     NvSciColor_FloatISP_Bayer16BCCR,
     NvSciColor_FloatISP_Bayer16CRBC,
     NvSciColor_FloatISP_Bayer16CBRC,
     NvSciColor_FloatISP_Bayer16RCCC,
     NvSciColor_FloatISP_Bayer16CCCR,
     NvSciColor_FloatISP_Bayer16CRCC,
     NvSciColor_FloatISP_Bayer16CCRC,
     NvSciColor_X12Bayer20CCCC,
     NvSciColor_X12Bayer20BGGR,
     NvSciColor_X12Bayer20RGGB,
     NvSciColor_X12Bayer20GRBG,
     NvSciColor_X12Bayer20GBRG,
     NvSciColor_X12Bayer20RCCB,
     NvSciColor_X12Bayer20BCCR,
     NvSciColor_X12Bayer20CRBC,
     NvSciColor_X12Bayer20CBRC,
     NvSciColor_X12Bayer20RCCC,
     NvSciColor_X12Bayer20CCCR,
     NvSciColor_X12Bayer20CRCC,
     NvSciColor_X12Bayer20CCRC,
     NvSciColor_Signed_X12Bayer20CCCC,
     /*
      * Note: This is currently not supported, and setting this attribute key
      * will fail.
      */
     NvSciColor_Signed_X12Bayer20GBRG,

     /* Semiplanar formats */
     /* Bit ordering for little endian machine is as follows
      * for NvSciColor_V8U8
      * pattern  VVVVVVVV UUUUUUUU
      *          +------+ +------+
      * bit      15     8 7      0
      *          +---------------+
      * pixel            0
      *
      * for NvSciColor_U8_V8
      * pattern  VVVVVVVV UUUUUUUU
      *          +------+ +------+
      * bit      15     8 7      0
      *          +---------------+
      * pixel            0
      *
      * for NvSciColor_V10U10
      * pattern  VVVVVVVV VV****** UUUUUUUU UU******
      *          +------+ +------+ +------+ +------+
      * bit      31    24 23    16 15     8 7      0
      *          +---------------------------------+
      * pixel                     0
      *
      * '*' means undefined bit value
      */
     NvSciColor_U8V8,
     NvSciColor_U8_V8,
     NvSciColor_V8U8,
     NvSciColor_V8_U8,
     NvSciColor_U10V10,
     NvSciColor_V10U10,
     NvSciColor_U12V12,
     NvSciColor_V12U12,
     NvSciColor_U16V16,
     NvSciColor_V16U16,

     /* PLANAR formats */
     /* Bit ordering for little endian machine is as follows
      * for NvSciColor_Y12
      * pattern  YYYYYYYY YYYY****
      *          +------+ +------+
      * bit      15     8 7      0
      *          +---------------+
      * pixel            0
      * '*' means undefined bit value
      */
     NvSciColor_Y8,
     NvSciColor_Y10,
     NvSciColor_Y12,
     NvSciColor_Y16,
     NvSciColor_U8,
     NvSciColor_V8,
     NvSciColor_U10,
     NvSciColor_V10,
     NvSciColor_U12,
     NvSciColor_V12,
     NvSciColor_U16,
     NvSciColor_V16,

     /* Packed YUV formats */
     /* NvSciColor_Y8U8Y8V8, NvSciColor_Y8V8Y8U8, NvSciColor_U8Y8V8Y8, NvSciColor_V8Y8U8Y8 follow
      * different component order than generic word representation.
      *
      * Bit ordering for little endian machine is as follows
      * for NvSciColor_U8Y8V8Y8
      * pattern  YYYYYYYY VVVVVVVV YYYYYYYY UUUUUUUU
      *          +------+ +------+ +------+ +------+
      * bit      31    24 23    16 15     8 7      0
      *          +---------------------------------+
      * pixel                     0
      *
      * for NvSciColor_A16Y16U16V16
      * pattern  AAAAAAAAAAAAAAAA YYYYYYYYYYYYYYYY UUUUUUUUUUUUUUUU VVVVVVVVVVVVVVVV
      *          +--------------+ +--------------+ +--------------+ +--------------+
      * bit      63            48 47            32 31            16 15             0
      *          +-----------------------------------------------------------------+
      * pixel                                     0
      *
      */
     NvSciColor_A8Y8U8V8,
     NvSciColor_Y8U8Y8V8,
     NvSciColor_Y8V8Y8U8,
     NvSciColor_U8Y8V8Y8,
     NvSciColor_V8Y8U8Y8,
     NvSciColor_A16Y16U16V16,

     /* RGBA PACKED */
     /* Bit ordering for little endian machine is as follows
      * for NvSciColor_A2R10G10B10
      * pattern  AARRRRRR RRRRGGGG GGGGGGBB BBBBBBBB
      *          +------+ +------+ +------+ +------+
      * bit      31    24 23    16 15     8 7      0
      *          +---------------------------------+
      * pixel                     0
      *
      */
     NvSciColor_A8,
     NvSciColor_Signed_A8,
     NvSciColor_B8G8R8A8,
     NvSciColor_A8R8G8B8,
     NvSciColor_A8B8G8R8,
     NvSciColor_A2R10G10B10,
     NvSciColor_A16,
     NvSciColor_Signed_A16,
     NvSciColor_Signed_R16G16,
     NvSciColor_A16B16G16R16,
     NvSciColor_Signed_A16B16G16R16,
     NvSciColor_Float_A16B16G16R16,
     NvSciColor_A32,
     NvSciColor_Signed_A32,
     NvSciColor_Float_A16,

     /* 10-bit 4x4 RGB-IR Bayer formats */
     /* Bit ordering for little endian machine is as follows
      * for NvSciColor_X6Bayer10BGGI_RGGI
      * pattern  GGGGGGGG GG****** RRRRRRRR RR****** GGGGGGGG GG****** BBBBBBBB BB******
      *          +------+ +------+ +------+ +------+ +------+ +------+ +------+ +------+
      * bit      63    56 55    48 47    40 39    32 31    24 23    16 15     8 7      0
      *          +---------------+ +---------------+ +---------------+ +---------------+
      * pixel            3                 2                 1                 0
      *          +---------------------------------------------------------------------+
      * pitch                                       0
      *
      * pattern  IIIIIIII II****** GGGGGGGG GG****** IIIIIIII II****** GGGGGGGG GG******
      *          +------+ +------+ +------+ +------+ +------+ +------+ +------+ +------+
      * bit      63    56 55    48 47    40 39    32 31    24 23    16 15     8 7      0
      *          +---------------+ +---------------+ +---------------+ +---------------+
      * pixel            3                 2                 1                 0
      *          +---------------------------------------------------------------------+
      * pitch                                       1
      *
      * pattern  GGGGGGGG GG****** BBBBBBBB BB****** GGGGGGGG GG****** RRRRRRRR RR******
      *          +------+ +------+ +------+ +------+ +------+ +------+ +------+ +------+
      * bit      63    56 55    48 47    40 39    32 31    24 23    16 15     8 7      0
      *          +---------------+ +---------------+ +---------------+ +---------------+
      * pixel            3                 2                 1                 0
      *          +---------------------------------------------------------------------+
      * pitch                                       2
      *
      * pattern  IIIIIIII II****** GGGGGGGG GG****** IIIIIIII II****** GGGGGGGG GG******
      *          +------+ +------+ +------+ +------+ +------+ +------+ +------+ +------+
      * bit      63    56 55    48 47    40 39    32 31    24 23    16 15     8 7      0
      *          +---------------+ +---------------+ +---------------+ +---------------+
      * pixel            3                 2                 1                 0
      *          +---------------------------------------------------------------------+
      * pitch                                       3
      *
      * '*' means undefined bit value
      */
     NvSciColor_X6Bayer10BGGI_RGGI,
     NvSciColor_X6Bayer10GBIG_GRIG,
     NvSciColor_X6Bayer10GIBG_GIRG,
     NvSciColor_X6Bayer10IGGB_IGGR,
     NvSciColor_X6Bayer10RGGI_BGGI,
     NvSciColor_X6Bayer10GRIG_GBIG,
     NvSciColor_X6Bayer10GIRG_GIBG,
     NvSciColor_X6Bayer10IGGR_IGGB,

     /* 12-bit 4x4 RGB-IR Bayer formats */
     NvSciColor_X4Bayer12BGGI_RGGI,
     NvSciColor_X4Bayer12GBIG_GRIG,
     NvSciColor_X4Bayer12GIBG_GIRG,
     NvSciColor_X4Bayer12IGGB_IGGR,
     NvSciColor_X4Bayer12RGGI_BGGI,
     NvSciColor_X4Bayer12GRIG_GBIG,
     NvSciColor_X4Bayer12GIRG_GIBG,
     NvSciColor_X4Bayer12IGGR_IGGB,

     /* 14-bit 4x4 RGB-IR Bayer formats */
     NvSciColor_X2Bayer14BGGI_RGGI,
     NvSciColor_X2Bayer14GBIG_GRIG,
     NvSciColor_X2Bayer14GIBG_GIRG,
     NvSciColor_X2Bayer14IGGB_IGGR,
     NvSciColor_X2Bayer14RGGI_BGGI,
     NvSciColor_X2Bayer14GRIG_GBIG,
     NvSciColor_X2Bayer14GIRG_GIBG,
     NvSciColor_X2Bayer14IGGR_IGGB,

     /* 16-bit 4x4 RGB-IR Bayer formats */
     NvSciColor_Bayer16BGGI_RGGI,
     NvSciColor_Bayer16GBIG_GRIG,
     NvSciColor_Bayer16GIBG_GIRG,
     NvSciColor_Bayer16IGGB_IGGR,
     NvSciColor_Bayer16RGGI_BGGI,
     NvSciColor_Bayer16GRIG_GBIG,
     NvSciColor_Bayer16GIRG_GIBG,
     NvSciColor_Bayer16IGGR_IGGB,

     /* Right-justified Bayer RAW format */
     /* Bit ordering for little endian machine is as follows
      * for NvSciColor_X4Bayer12RGGB_RJ
      * pattern  ****GGGG GGGGGGGG ****RRRR RRRRRRRR
      *          +------+ +------+ +------+ +------+
      * bit      31    24 23    16 15     8 7      0
      *          +---------------+ +---------------+
      * pixel            1                 0
      *          +---------------------------------+
      * pitch                     0
      *
      * pattern  ****BBBB BBBBBBBB ****GGGG GGGGGGGG
      *          +------+ +------+ +------+ +------+
      * bit      31    24 23    16 15     8 7      0
      *          +---------------+ +---------------+
      * pixel            1                 0
      *          +---------------------------------+
      * pitch                     1
      *
      * '*' means undefined bit value
      */
     NvSciColor_X4Bayer12RGGB_RJ,

     /* RGB PLANAR */
     NvSciColor_R8,
     NvSciColor_G8,
     NvSciColor_B8,

     NvSciColor_UpperBound
 } NvSciBufAttrValColorFmt;

 typedef enum {
     NvSciColorStd_SRGB,
     NvSciColorStd_REC601_SR,
     NvSciColorStd_REC601_ER,
     NvSciColorStd_REC709_SR,
     NvSciColorStd_REC709_ER,
     NvSciColorStd_REC2020_RGB,
     NvSciColorStd_REC2020_SR,
     NvSciColorStd_REC2020_ER,
     NvSciColorStd_YcCbcCrc_SR,
     NvSciColorStd_YcCbcCrc_ER,
     NvSciColorStd_SENSOR_RGBA,
     NvSciColorStd_REQ2020PQ_ER,
 } NvSciBufAttrValColorStd;

 typedef enum {
     NvSciSurfType_YUV,
     NvSciSurfType_RGBA,
     NvSciSurfType_RAW,
     NvSciSurfType_MaxValid,
 } NvSciBufSurfType;

 typedef enum {
     NvSciSurfMemLayout_Packed,
     NvSciSurfMemLayout_SemiPlanar,
     NvSciSurfMemLayout_Planar,
     NvSciSurfMemLayout_MaxValid,
 } NvSciBufSurfMemLayout;

 typedef enum {
     NvSciSurfSampleType_420,
     NvSciSurfSampleType_422,
     NvSciSurfSampleType_444,
     NvSciSurfSampleType_422R,
     NvSciSurfSampleType_400,
     NvSciSurfSampleType_MaxValid,
 } NvSciBufSurfSampleType;

 typedef enum {
     NvSciSurfBPC_Layout_16_8_8,
     NvSciSurfBPC_Layout_10_8_8,
     NvSciSurfBPC_8,
     NvSciSurfBPC_10,
     NvSciSurfBPC_12,
     NvSciSurfBPC_14,
     NvSciSurfBPC_16,
     NvSciSurfBPC_MaxValid,
 } NvSciBufSurfBPC;

 typedef enum {
     NvSciSurfComponentOrder_YUV,
     NvSciSurfComponentOrder_YVU,
     NvSciSurfComponentOrder_Luma,
     NvSciSurfComponentOrder_MaxValid,
 } NvSciBufSurfComponentOrder;

 typedef enum {
     NvSciDataType_Int4,
     NvSciDataType_Uint4,
     NvSciDataType_Int8,
     NvSciDataType_Uint8,
     NvSciDataType_Int16,
     NvSciDataType_Uint16,
     NvSciDataType_Int32,
     NvSciDataType_Uint32,
     NvSciDataType_Float16,
     NvSciDataType_Float32,
     NvSciDataType_FloatISP,
     NvSciDataType_Bool,
     NvSciDataType_UpperBound
 } NvSciBufAttrValDataType;

 typedef enum {
     NvSciBufCompressionType_None,

     NvSciBufCompressionType_GenericCompressible,
 } NvSciBufCompressionType;

 typedef struct {
     uint8_t bytes[16];
 } NvSciRmGpuId;

 typedef struct {
     NvSciRmGpuId gpuId;

     bool cacheability;
 } NvSciBufAttrValGpuCache;

 typedef struct {
     NvSciRmGpuId gpuId;

     NvSciBufCompressionType compressionType;
 } NvSciBufAttrValGpuCompression;

 typedef struct {
     uint64_t x0;
     uint64_t y0;
     uint64_t x1;
     uint64_t y1;
 } NvSciBufRect;

 typedef enum {
     NvSciBufHwEngName_Invalid   = 0,
     NvSciBufHwEngName_Display   = 4,
     NvSciBufHwEngName_Isp       = 11,
     NvSciBufHwEngName_Vi        = 12,
     NvSciBufHwEngName_Csi       = 30,
     NvSciBufHwEngName_Vic       = 106,
     NvSciBufHwEngName_Gpu       = 107,
     NvSciBufHwEngName_MSENC     = 109,
     NvSciBufHwEngName_NVDEC     = 117,
     NvSciBufHwEngName_NVJPG     = 118,
     NvSciBufHwEngName_PVA       = 121,
     NvSciBufHwEngName_DLA       = 122,
     NvSciBufHwEngName_PCIe      = 123,
     NvSciBufHwEngName_OFA       = 124,
     NvSciBufHwEngName_NPM       = 125,
     NvSciBufHwEngName_Num       = 126
 } NvSciBufHwEngName;

 typedef enum {
     NvSciBufPlatformName_LowerBound,
     NvSciBufPlatformName_Orin,
     NvSciBufPlatformName_PG189,
     NvSciBufPlatformName_PG199,
     NvSciBufPlatformName_UpperBound
 } NvSciBufPlatformName;

 PACK_BUF(typedef struct {
     NvSciBufHwEngName engName;
     NvSciBufPlatformName platName;
 }) NvSciBufPeerHwEngine;

 typedef struct NvSciBufModuleRec* NvSciBufModule;

 typedef struct {
       NvSciBufAttrKey key;

       const void* value;

       size_t len;
 } NvSciBufAttrKeyValuePair;

 typedef struct NvSciBufObjRefRec* NvSciBufObj;

 typedef const struct NvSciBufObjRefRec* NvSciBufObjConst;


 typedef struct NvSciBufAttrListRec* NvSciBufAttrList;

 PACK_BUF(typedef struct {
       uint64_t data[NVSCIBUF_EXPORT_DESC_SIZE];
 }) NvSciBufObjIpcExportDescriptor;

 PACK_BUF(typedef struct {
     uint32_t socID;
     uint32_t vmID;
     uint64_t reserved;
 }) NvSciBufPeerLocationInfo;

 NvSciError NvSciBufAttrListCreate(
     NvSciBufModule module,
     NvSciBufAttrList* newAttrList);

 void NvSciBufAttrListFree(
     NvSciBufAttrList attrList);

 NvSciError NvSciBufAttrListSetAttrs(
     NvSciBufAttrList attrList,
     NvSciBufAttrKeyValuePair* pairArray,
     size_t pairCount);

 size_t NvSciBufAttrListGetSlotCount(
     NvSciBufAttrList attrList);

 NvSciError NvSciBufAttrListGetAttrs(
     NvSciBufAttrList attrList,
     NvSciBufAttrKeyValuePair* pairArray,
     size_t pairCount);

 NvSciError NvSciBufAttrListSlotGetAttrs(
     NvSciBufAttrList attrList,
     size_t slotIndex,
     NvSciBufAttrKeyValuePair* pairArray,
     size_t pairCount);

 #if (NV_IS_SAFETY == 0)

 NvSciError NvSciBufAttrListDebugDump(
     NvSciBufAttrList attrList,
     void** buf,
     size_t* len);
 #endif

 #if (NV_IS_SAFETY == 0)

 #else

 #endif

 #if (NV_IS_SAFETY == 0)

 #endif

 NvSciError NvSciBufAttrListReconcile(
     const NvSciBufAttrList inputArray[],
     size_t inputCount,
     NvSciBufAttrList* newReconciledAttrList,
     NvSciBufAttrList* newConflictList);

 NvSciError NvSciBufAttrListClone(
     NvSciBufAttrList origAttrList,
     NvSciBufAttrList* newAttrList);

 NvSciError NvSciBufAttrListAppendUnreconciled(
     const NvSciBufAttrList inputUnreconciledAttrListArray[],
     size_t inputUnreconciledAttrListCount,
     NvSciBufAttrList* newUnreconciledAttrList);

 NvSciError NvSciBufAttrListIsReconciled(
     NvSciBufAttrList attrList,
     bool* isReconciled);

 NvSciError NvSciBufAttrListValidateReconciled(
     NvSciBufAttrList reconciledAttrList,
     const NvSciBufAttrList unreconciledAttrListArray[],
     size_t unreconciledAttrListCount,
     bool* isReconcileListValid);

 NvSciError NvSciBufObjDup(
     NvSciBufObj bufObj,
     NvSciBufObj* dupObj);

 #if (NV_IS_SAFETY == 0)

 #else

 #endif

 #if (NV_IS_SAFETY == 0)

 #endif

 NvSciError NvSciBufAttrListReconcileAndObjAlloc(
     const NvSciBufAttrList attrListArray[],
     size_t attrListCount,
     NvSciBufObj* bufObj,
     NvSciBufAttrList* newConflictList);

 void NvSciBufObjFree(
     NvSciBufObj bufObj);

 NvSciError NvSciBufObjGetAttrList(
     NvSciBufObj bufObj,
     NvSciBufAttrList* bufAttrList);

 NvSciError NvSciBufObjGetCpuPtr(
     NvSciBufObj bufObj,
     void**  ptr);

 NvSciError NvSciBufObjGetConstCpuPtr(
     NvSciBufObj bufObj,
     const void**  ptr);

 NvSciError NvSciBufObjFlushCpuCacheRange(
     NvSciBufObj bufObj,
     uint64_t offset,
     uint64_t len);

 NvSciError NvSciBufObjAlloc(
     NvSciBufAttrList reconciledAttrList,
     NvSciBufObj* bufObj);

 #if (NV_IS_SAFETY == 0)

 #endif

 NvSciError NvSciBufObjDupWithReducePerm(
     NvSciBufObj bufObj,
     NvSciBufAttrValAccessPerm reducedPerm,
     NvSciBufObj* newBufObj);

 NvSciError NvSciBufObjGetPixels(
     NvSciBufObj bufObj,
     const NvSciBufRect* rect,
     void** dstPtrs,
     const uint32_t* dstPtrSizes,
     const uint32_t* dstPitches);

 NvSciError NvSciBufObjPutPixels(
     NvSciBufObj bufObj,
     const NvSciBufRect* rect,
     const void** srcPtrs,
     const uint32_t* srcPtrSizes,
     const uint32_t* srcPitches);

 NvSciError NvSciBufIpcExportAttrListAndObj(
     NvSciBufObj bufObj,
     NvSciBufAttrValAccessPerm permissions,
     NvSciIpcEndpoint ipcEndpoint,
     void** attrListAndObjDesc,
     size_t* attrListAndObjDescSize);

 NvSciError NvSciBufIpcImportAttrListAndObj(
     NvSciBufModule module,
     NvSciIpcEndpoint ipcEndpoint,
     const void* attrListAndObjDesc,
     size_t attrListAndObjDescSize,
     const NvSciBufAttrList attrList[],
     size_t count,
     NvSciBufAttrValAccessPerm minPermissions,
     int64_t timeoutUs,
     NvSciBufObj* bufObj);

 void NvSciBufAttrListAndObjFreeDesc(
     void* attrListAndObjDescBuf);

 NvSciError NvSciBufObjIpcExport(
     NvSciBufObj bufObj,
     NvSciBufAttrValAccessPerm accPerm,
     NvSciIpcEndpoint ipcEndpoint,
     NvSciBufObjIpcExportDescriptor* exportData);

 #if defined (BACKEND_RESMAN)
 #if (BACKEND_RESMAN)

 #endif
 #endif

 NvSciError NvSciBufObjIpcImport(
     NvSciIpcEndpoint ipcEndpoint,
     const NvSciBufObjIpcExportDescriptor* desc,
     NvSciBufAttrList reconciledAttrList,
     NvSciBufAttrValAccessPerm minPermissions,
     int64_t timeoutUs,
     NvSciBufObj* bufObj);

 NvSciError NvSciBufObjAttachPeer(
     NvSciBufObj bufObj,
     const NvSciBufAttrList inputArray[],
     size_t inputCount);

 NvSciError NvSciBufAttrListIpcExportUnreconciled(
     const NvSciBufAttrList unreconciledAttrListArray[],
     size_t unreconciledAttrListCount,
     NvSciIpcEndpoint ipcEndpoint,
     void** descBuf,
     size_t* descLen);

 NvSciError NvSciBufAttrListIpcExportReconciled(
     NvSciBufAttrList reconciledAttrList,
     NvSciIpcEndpoint ipcEndpoint,
     void** descBuf,
     size_t* descLen);

 NvSciError NvSciBufAttrListIpcImportUnreconciled(
     NvSciBufModule module,
     NvSciIpcEndpoint ipcEndpoint,
     const void* descBuf,
     size_t descLen,
     NvSciBufAttrList* importedUnreconciledAttrList);

 NvSciError NvSciBufAttrListIpcImportReconciled(
     NvSciBufModule module,
     NvSciIpcEndpoint ipcEndpoint,
     const void* descBuf,
     size_t descLen,
     const NvSciBufAttrList inputUnreconciledAttrListArray[],
     size_t inputUnreconciledAttrListCount,
     NvSciBufAttrList* importedReconciledAttrList);


 void NvSciBufAttrListFreeDesc(
     void* descBuf);

 NvSciError NvSciBufModuleOpen(
     NvSciBufModule* newModule);

 void NvSciBufModuleClose(
     NvSciBufModule module);

 #if (NV_IS_SAFETY == 0)

 #endif

 #if (NV_IS_SAFETY == 0)

 #endif

 NvSciError NvSciBufCheckVersionCompatibility(
     uint32_t majorVer,
     uint32_t minorVer,
     bool* isCompatible);

 #if (NV_IS_SAFETY == 0)

 #else

 #endif

 NvSciError NvSciBufObjValidate(
     NvSciBufObj bufObj);

 NvSciError NvSciBufAttrListValidateReconciledAgainstAttrs(
     const NvSciBufAttrList reconciledAttrList,
     const NvSciBufAttrKeyValuePair* pairArray,
     const size_t pairCount);

 NvSciError NvSciBufObjGetMaxPerm(
     const NvSciBufObj bufObj,
     const NvSciIpcEndpoint ipcEndpoint,
     NvSciBufAttrValAccessPerm* accPerm);

 #if defined(__cplusplus)
 }
 #endif // __cplusplus

 #endif /* INCLUDED_NVSCIBUF_H */
