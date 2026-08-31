 /*
  * Copyright (c) 2019-2023, NVIDIA CORPORATION. All rights reserved.
  *
  * NVIDIA Corporation and its licensors retain all intellectual property
  * and proprietary rights in and to this software, related documentation
  * and any modifications thereto.  Any use, reproduction, disclosure or
  * distribution of this software and related documentation without an express
  * license agreement from NVIDIA Corporation is strictly prohibited.
  */

 #ifndef INCLUDED_NVSCI_ERROR_H
 #define INCLUDED_NVSCI_ERROR_H

 #ifdef __cplusplus
 extern "C" {
 #endif

 typedef enum {
     /* Range 0x00000000 - 0x00FFFFFF : Common errors
      * This range is used for errors common to all NvSci libraries. */

     NvSciError_Success                  = 0x00000000,

     NvSciError_Unknown                  = 0x00000001,

     /* Generic errors */
     NvSciError_NotImplemented           = 0x00000010,
     NvSciError_NotSupported             = 0x00000011,
     NvSciError_AccessDenied             = 0x00000020,
     NvSciError_NotPermitted             = 0x00000021,
     NvSciError_InvalidState             = 0x00000022,
     NvSciError_InvalidOperation         = 0x00000023,
     NvSciError_NotInitialized           = 0x00000024,
     NvSciError_AlreadyInUse             = 0x00000025,
     NvSciError_AlreadyDone              = 0x00000026,
     NvSciError_NotYetAvailable          = 0x00000027,
     NvSciError_NoLongerAvailable        = 0x00000028,
     NvSciError_InsufficientMemory       = 0x00000030,
     NvSciError_InsufficientResource     = 0x00000031,
     NvSciError_ResourceError            = 0x00000032,

     /* Function parameter errors */
     NvSciError_BadParameter             = 0x00000100,
     NvSciError_BadAddress               = 0x00000101,
     NvSciError_TooBig                   = 0x00000102,
     NvSciError_Overflow                 = 0x00000103,
     NvSciError_InconsistentData         = 0x00000104,
     NvSciError_InsufficientData         = 0x00000105,
     NvSciError_IndexOutOfRange          = 0x00000106,
     NvSciError_ValueOutOfRange          = 0x00000107,
     NvSciError_Revalidation_Success     = 0x00000108,

     /* Timing/temporary errors */
     NvSciError_Timeout                  = 0x00000200,
     NvSciError_TryItAgain               = 0x00000201,
     NvSciError_Busy                     = 0x00000202,
     NvSciError_InterruptedCall          = 0x00000203,

     /* Device errors */
     NvSciError_NoSuchDevice             = 0x00001000,
     NvSciError_NoSpace                  = 0x00001001,
     NvSciError_NoSuchDevAddr            = 0x00001002,
     NvSciError_IO                       = 0x00001003,
     NvSciError_InvalidIoctlNum          = 0x00001004,

     /* File system errors */
     NvSciError_NoSuchEntry              = 0x00001100,
     NvSciError_BadFileDesc              = 0x00001101,
     NvSciError_CorruptedFileSys         = 0x00001102,
     NvSciError_FileExists               = 0x00001103,
     NvSciError_IsDirectory              = 0x00001104,
     NvSciError_ReadOnlyFileSys          = 0x00001105,
     NvSciError_TextFileBusy             = 0x00001106,
     NvSciError_FileNameTooLong          = 0x00001107,
     NvSciError_FileTooBig               = 0x00001108,
     NvSciError_TooManySymbolLinks       = 0x00001109,
     NvSciError_TooManyOpenFiles         = 0x0000110A,
     NvSciError_FileTableOverflow        = 0x0000110B,
     NvSciError_EndOfFile                = 0x0000110C,


     /* Communication errors */
     NvSciError_ConnectionReset          = 0x00001200,
     NvSciError_AlreadyInProgress        = 0x00001201,
     NvSciError_NoData                   = 0x00001202,
     NvSciError_NoDesiredMessage         = 0x00001203,
     NvSciError_MessageSize              = 0x00001204,
     NvSciError_NoRemote                 = 0x00001205,

     /* Process/thread errors */
     NvSciError_NoSuchProcess            = 0x00002000,

     /* Mutex errors */
     NvSciError_MutexNotRecoverable      = 0x00002100,
     NvSciError_LockOwnerDead            = 0x00002101,
     NvSciError_ResourceDeadlock         = 0x00002102,

     /* NvSci attribute list errors */
     NvSciError_ReconciliationFailed     = 0x00010100,
     NvSciError_AttrListValidationFailed = 0x00010101,
     NvSciError_ObjValidationFailed      = 0x00010102,



     NvSciError_CommonEnd                = 0x00FFFFFF,


     /* Range 0x01000000 - 0x01FFFFFF : NvSciBuf errors */
     NvSciError_NvSciBufUnknown          = 0x01000000,
     NvSciError_NvSciBufEnd              = 0x01FFFFFF,


     /* Range 0x02000000 - 0x02FFFFFF : NvSciSync errors */
     NvSciError_NvSciSyncUnknown         = 0x02000000,
     NvSciError_UnsupportedConfig        = 0x02000001,
     NvSciError_ClearedFence             = 0x02000002,
     /* End of range for NvScSync errors */
     NvSciError_NvSciSyncEnd             = 0x02FFFFFF,


     /* Range 0x03000000 - 0x03FFFFFF : NvSciStream errors */

     NvSciError_NvSciStreamUnknown       = 0x03000000,
     NvSciError_StreamInternalError      = 0x03000001,
     NvSciError_StreamBadBlock           = 0x03000100,
     NvSciError_StreamBadPacket          = 0x03000101,
     NvSciError_StreamBadCookie          = 0x03000102,
     NvSciError_StreamNotConnected       = 0x03000200,
     NvSciError_StreamNotSetupPhase      = 0x03000201,
     NvSciError_StreamNotSafetyPhase     = 0x03000202,
     NvSciError_NoStreamPacket           = 0x03001000,
     NvSciError_StreamPacketInaccessible = 0x03001001,
     NvSciError_StreamPacketDeleted      = 0x03001002,
     NvSciError_StreamInfoNotProvided    = 0x03003000,

     NvSciError_StreamLockFailed         = 0x03400000,

     NvSciError_StreamBadSrcIndex        = 0x03800000,
     NvSciError_StreamBadDstIndex        = 0x03800001,

     NvSciError_NvSciStreamEnd           = 0x03FFFFFF,


     /* Range 0x04000000 - 0x04FFFFFF : NvSciIpc errors */
     NvSciError_NvSciIpcUnknown          = 0x04000000,
     NvSciError_PcieUncorrectableFatal   = 0x04000001,
     NvSciError_PcieUncorrectableNonFatal= 0x04000002,
     NvSciError_PcieEdmaTransferErr      = 0x04000003,
     NvSciError_PcieValidationError      = 0x04000004,
     NvSciError_NvSciIpcEnd              = 0x04FFFFFF,


     /* Range 0x05000000 - 0x05FFFFFF : NvSciEvent errors */
     NvSciError_NvSciEventUnknown        = 0x05000000,
     NvSciError_NvSciEventEnd            = 0x05FFFFFF,

 } NvSciError;

 #ifdef __cplusplus
 }
 #endif

 #endif /* INCLUDED_NVSCI_ERROR_H */
