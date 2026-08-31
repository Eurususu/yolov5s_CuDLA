 /*
  * Copyright (c) 2019-2023, NVIDIA CORPORATION. All rights reserved.
  *
  * NVIDIA Corporation and its licensors retain all intellectual property
  * and proprietary rights in and to this software, related documentation
  * and any modifications thereto.  Any use, reproduction, disclosure or
  * distribution of this software and related documentation without an express
  * license agreement from NVIDIA Corporation is strictly prohibited.
  */
 #ifndef INCLUDED_NVSCISYNC_H
 #define INCLUDED_NVSCISYNC_H

 #if !defined (__cplusplus)
 #include <stddef.h>
 #include <stdbool.h>
 #include <stdint.h>
 #endif

 #include "nvscierror.h"
 #include "nvsciipc.h"
 #include "nvscibuf.h"

 #if defined (__cplusplus)
 extern "C"
 {
 #endif

 #if defined __GNUC__
     #define PACK_SYNC( __Declaration__, ... ) __Declaration__,##__VA_ARGS__ __attribute__((packed))
 #else
     #define PACK_SYNC( __Declaration__, ... ) __pragma(pack(push, 1)) __Declaration__,##__VA_ARGS__ __pragma(pack(pop))
 #endif

 static const uint32_t NvSciSyncMajorVersion = 2U;

 static const uint32_t NvSciSyncMinorVersion = 8U;

 static const int64_t NvSciSyncFenceMaxTimeout = (0x7fffffffffffffff / 1000);

 typedef struct NvSciSyncModuleRec* NvSciSyncModule;

 typedef struct NvSciSyncCpuWaitContextRec* NvSciSyncCpuWaitContext;

 typedef struct NvSciSyncFence {
     uint64_t payload[6];
 } NvSciSyncFence;

 static const NvSciSyncFence NvSciSyncFenceInitializer = {{0U}};

 typedef struct {
     uint64_t payload[7];
 } NvSciSyncFenceIpcExportDescriptor;

 typedef struct {
     uint64_t payload[128];
 } NvSciSyncObjIpcExportDescriptor;

 typedef struct NvSciSyncObjRec* NvSciSyncObj;

 typedef const struct NvSciSyncObjRec* NvSciSyncObjConst;

 typedef struct NvSciSyncAttrListRec* NvSciSyncAttrList;

 typedef uint64_t NvSciSyncAccessPerm;

 #if defined(__cplusplus)
     #define NvSciSyncAccessPerm_WaitOnly (static_cast<uint64_t>(1U) << 0U)
 #else
     #define NvSciSyncAccessPerm_WaitOnly ((uint64_t)1U << 0U)
 #endif

 #if defined(__cplusplus)
     #define NvSciSyncAccessPerm_SignalOnly (static_cast<uint64_t>(1U) << 1U)
 #else
     #define NvSciSyncAccessPerm_SignalOnly ((uint64_t)1U << 1U)
 #endif

 #define NvSciSyncAccessPerm_WaitSignal (NvSciSyncAccessPerm_WaitOnly | NvSciSyncAccessPerm_SignalOnly)

 #if defined(__cplusplus)
     #define NvSciSyncAccessPerm_Auto (static_cast<uint64_t>(1U) << 63U)
 #else
     #define NvSciSyncAccessPerm_Auto ((uint64_t)1U << 63U)
 #endif

 typedef enum {
     NvSciSyncTaskStatus_Success = 0U,
     NvSciSyncTaskStatus_Failure = 1U,
     NvSciSyncTaskStatus_Invalid = UINT16_MAX,
 } NvSciSyncTaskStatusVal;

 PACK_SYNC(typedef struct {
     uint64_t timestamp;
     uint32_t statusEngine;
     uint16_t subframe;
     uint16_t status;
 }) NvSciSyncTaskStatus;

 enum NvSciSyncAttrValPrimitiveTypeRec {
     NvSciSyncAttrValPrimitiveType_LowerBound,
     NvSciSyncAttrValPrimitiveType_Syncpoint,
     NvSciSyncAttrValPrimitiveType_SysmemSemaphore,
     NvSciSyncAttrValPrimitiveType_VidmemSemaphore,
     NvSciSyncAttrValPrimitiveType_SysmemSemaphorePayload64b,
     NvSciSyncAttrValPrimitiveType_VidmemSemaphorePayload64b,
     NvSciSyncAttrValPrimitiveType_UpperBound,
 };

 typedef enum NvSciSyncAttrValPrimitiveTypeRec NvSciSyncAttrValPrimitiveType;

 typedef enum {
     NvSciSyncAttrKey_LowerBound,
     NvSciSyncAttrKey_NeedCpuAccess,
     NvSciSyncAttrKey_RequiredPerm,
     NvSciSyncAttrKey_ActualPerm,
     NvSciSyncAttrKey_WaiterContextInsensitiveFenceExports,
     NvSciSyncAttrKey_WaiterRequireTimestamps,
     NvSciSyncAttrKey_RequireDeterministicFences,
     NvSciSyncAttrKey_NumTimestampSlots,
     NvSciSyncAttrKey_NumTaskStatusSlots,
     NvSciSyncAttrKey_MaxPrimitiveValue,
     NvSciSyncAttrKey_PrimitiveInfo,

     NvSciSyncAttrKey_PeerLocationInfo,
     NvSciSyncAttrKey_GpuId,
     NvSciSyncAttrKey_PeerHwEngineArray,
     NvSciSyncAttrKey_UpperBound,
 } NvSciSyncAttrKey;

 typedef struct {
     NvSciSyncAttrKey attrKey;
     const void* value;
     size_t len;
 } NvSciSyncAttrKeyValuePair;

 typedef enum {
     NvSciSyncWaitMode_Default = 0U,
     NvSciSyncWaitMode_BusyWithYield = 1U,
     NvSciSyncWaitMode_BusyNoYield = 2U,
     NvSciSyncWaitMode_Blocking = 3U,
 } NvSciSyncWaitMode;

 NvSciError NvSciSyncModuleOpen(
     NvSciSyncModule* newModule);

 void NvSciSyncModuleClose(
     NvSciSyncModule module);

 NvSciError NvSciSyncCpuWaitContextAlloc(
     NvSciSyncModule module,
     NvSciSyncCpuWaitContext* newContext);

 void NvSciSyncCpuWaitContextFree(
     NvSciSyncCpuWaitContext context);

 NvSciError NvSciSyncAttrListCreate(
     NvSciSyncModule module,
     NvSciSyncAttrList* attrList);

 void NvSciSyncAttrListFree(
     NvSciSyncAttrList attrList);

 NvSciError NvSciSyncAttrListIsReconciled(
     NvSciSyncAttrList attrList,
     bool* isReconciled);

 NvSciError NvSciSyncAttrListValidateReconciled(
     NvSciSyncAttrList reconciledAttrList,
     const NvSciSyncAttrList inputUnreconciledAttrListArray[],
     size_t inputUnreconciledAttrListCount,
     bool* isReconciledListValid);

 NvSciError NvSciSyncAttrListSetAttrs(
     NvSciSyncAttrList attrList,
     const NvSciSyncAttrKeyValuePair* pairArray,
     size_t pairCount);

 NvSciError NvSciSyncAttrListGetAttrs(
     NvSciSyncAttrList attrList,
     NvSciSyncAttrKeyValuePair* pairArray,
     size_t pairCount);

 size_t NvSciSyncAttrListGetSlotCount(
     NvSciSyncAttrList attrList);

 NvSciError NvSciSyncAttrListAppendUnreconciled(
     const NvSciSyncAttrList inputUnreconciledAttrListArray[],
     size_t inputUnreconciledAttrListCount,
     NvSciSyncAttrList* newUnreconciledAttrList);

 NvSciError NvSciSyncAttrListClone(
     NvSciSyncAttrList origAttrList,
     NvSciSyncAttrList* newAttrList);

 NvSciError NvSciSyncAttrListSlotGetAttrs(
     NvSciSyncAttrList attrList,
     size_t slotIndex,
     NvSciSyncAttrKeyValuePair* pairArray,
     size_t pairCount);

 #if (NV_IS_SAFETY == 0)

 #endif

 #if (NV_IS_SAFETY == 0)

 #else

 #endif

 #if (NV_IS_SAFETY == 0)

 #endif

 NvSciError NvSciSyncAttrListReconcile(
     const NvSciSyncAttrList inputArray[],
     size_t inputCount,
     NvSciSyncAttrList* newReconciledList,
     NvSciSyncAttrList* newConflictList);

 #if (NV_IS_SAFETY == 0)

 NvSciError NvSciSyncAttrListDebugDump(
     NvSciSyncAttrList attrList,
     void** buf,
     size_t* len);
 #endif

 #if (NV_IS_SAFETY == 0)

 #endif

 NvSciError NvSciSyncAttrListIpcExportUnreconciled(
     const NvSciSyncAttrList unreconciledAttrListArray[],
     size_t unreconciledAttrListCount,
     NvSciIpcEndpoint ipcEndpoint,
     void** descBuf,
     size_t* descLen);

 NvSciError NvSciSyncAttrListIpcExportReconciled(
     const NvSciSyncAttrList reconciledAttrList,
     NvSciIpcEndpoint ipcEndpoint,
     void** descBuf,
     size_t* descLen);

 NvSciError NvSciSyncAttrListIpcImportUnreconciled(
     NvSciSyncModule module,
     NvSciIpcEndpoint ipcEndpoint,
     const void* descBuf,
     size_t descLen,
     NvSciSyncAttrList* importedUnreconciledAttrList);

 NvSciError NvSciSyncAttrListIpcImportReconciled(
     NvSciSyncModule module,
     NvSciIpcEndpoint ipcEndpoint,
     const void* descBuf,
     size_t descLen,
     const NvSciSyncAttrList inputUnreconciledAttrListArray[],
     size_t inputUnreconciledAttrListCount,
     NvSciSyncAttrList* importedReconciledAttrList);

 void NvSciSyncAttrListFreeDesc(
     void* descBuf);

 void NvSciSyncFenceClear(
     NvSciSyncFence* syncFence);

 NvSciError NvSciSyncFenceDup(
     const NvSciSyncFence* srcSyncFence,
     NvSciSyncFence* dstSyncFence);

 NvSciError NvSciSyncFenceExtractFence(
     const NvSciSyncFence* syncFence,
     uint64_t* id,
     uint64_t* value);

 NvSciError NvSciSyncFenceUpdateFence(
     NvSciSyncObj syncObj,
     uint64_t id,
     uint64_t value,
     NvSciSyncFence* syncFence);

 NvSciError NvSciSyncFenceAddTimestampSlot(
     NvSciSyncFence* syncFence,
     uint32_t timestampSlot);

 NvSciError NvSciSyncFenceAddTaskStatusSlot(
     NvSciSyncFence* syncFence,
     uint32_t taskStatusSlot);

 NvSciError NvSciSyncFenceExtractTimestampSlot(
     NvSciSyncFence* syncFence,
     uint32_t* timeStampSlot);

 NvSciError NvSciSyncObjAlloc(
     NvSciSyncAttrList reconciledList,
     NvSciSyncObj* syncObj);

 NvSciError NvSciSyncObjDup(
     NvSciSyncObj syncObj,
     NvSciSyncObj* dupObj);

 NvSciError NvSciSyncObjGetAttrList(
     NvSciSyncObj syncObj,
     NvSciSyncAttrList* syncAttrList);

 void NvSciSyncObjFree(
     NvSciSyncObj syncObj);

 #if (NV_L4T == 1)

 #endif

 NvSciError NvSciSyncObjIpcExport(
     NvSciSyncObj syncObj,
     NvSciSyncAccessPerm permissions,
     NvSciIpcEndpoint ipcEndpoint,
     NvSciSyncObjIpcExportDescriptor* desc);

 NvSciError NvSciSyncObjIpcImport(
     NvSciIpcEndpoint ipcEndpoint,
     const NvSciSyncObjIpcExportDescriptor* desc,
     NvSciSyncAttrList inputAttrList,
     NvSciSyncAccessPerm permissions,
     int64_t timeoutUs,
     NvSciSyncObj* syncObj);

 NvSciError NvSciSyncIpcExportFence(
     const NvSciSyncFence* syncFence,
     NvSciIpcEndpoint ipcEndpoint,
     NvSciSyncFenceIpcExportDescriptor* desc);

 NvSciError NvSciSyncIpcImportFence(
     NvSciSyncObj syncObj,
     const NvSciSyncFenceIpcExportDescriptor* desc,
     NvSciSyncFence* syncFence);

 #if (NV_IS_SAFETY == 0)

 #else

 #endif

 NvSciError NvSciSyncObjValidate(
     NvSciSyncObj syncObj);

 #if (NV_IS_SAFETY == 0)

 #endif

 NvSciError NvSciSyncObjGenerateFence(
     NvSciSyncObj syncObj,
     NvSciSyncFence* syncFence);

 #if (NV_IS_SAFETY == 0)

 #endif

 NvSciError NvSciSyncObjSignal(
     NvSciSyncObj syncObj);

 NvSciError NvSciSyncFenceWait(
     const NvSciSyncFence* syncFence,
     NvSciSyncCpuWaitContext context,
     int64_t timeoutUs);

 NvSciError NvSciSyncFenceGetTimestamp(
     const NvSciSyncFence* syncFence,
     uint64_t* timestampUS);

 NvSciError NvSciSyncFenceGetTaskStatus(
     const NvSciSyncFence* syncFence,
     NvSciSyncTaskStatus* taskStatus);

 /*
  * NvSciSync Utility functions
  */

 NvSciError NvSciSyncAttrListGetAttr(
     NvSciSyncAttrList attrList,
     NvSciSyncAttrKey key,
     const void** value,
     size_t* len);

 #if (NV_IS_SAFETY == 0)

 #else

 #endif

 #if (NV_IS_SAFETY == 0)

 #endif

 NvSciError NvSciSyncAttrListReconcileAndObjAlloc(
     const NvSciSyncAttrList inputArray[],
     size_t inputCount,
     NvSciSyncObj* syncObj,
     NvSciSyncAttrList* newConflictList);

 NvSciError NvSciSyncIpcExportAttrListAndObj(
     NvSciSyncObj syncObj,
     NvSciSyncAccessPerm permissions,
     NvSciIpcEndpoint ipcEndpoint,
     void** attrListAndObjDesc,
     size_t* attrListAndObjDescSize);

 void NvSciSyncAttrListAndObjFreeDesc(
     void* attrListAndObjDescBuf);

 NvSciError NvSciSyncIpcImportAttrListAndObj(
     NvSciSyncModule module,
     NvSciIpcEndpoint ipcEndpoint,
     const void* attrListAndObjDesc,
     size_t attrListAndObjDescSize,
     NvSciSyncAttrList const attrList[],
     size_t attrListCount,
     NvSciSyncAccessPerm minPermissions,
     int64_t timeoutUs,
     NvSciSyncObj* syncObj);

 #if (NV_IS_SAFETY == 0)

 #endif
 NvSciError NvSciSyncCheckVersionCompatibility(
     uint32_t majorVer,
     uint32_t minorVer,
     bool* isCompatible);

 NvSciError NvSciSyncObjAttachPeer(
     NvSciSyncObj syncObj,
     const NvSciSyncAttrList inputArray[],
     size_t inputCount);

 NvSciError NvSciSyncFillC2cAttrs(
     NvSciSyncAttrList unrecAttrList,
     NvSciSyncAccessPerm permissions);

 #if (NV_IS_SAFETY == 0)

 #else

 #endif

 NvSciError NvSciSyncAttrListValidateReconciledAgainstAttrs(
     NvSciSyncAttrList reconciledAttrList,
     const NvSciSyncAttrKeyValuePair* pairArray,
     size_t pairCount,
     NvSciSyncAccessPerm permissions);

 NvSciError NvSciSyncFenceWaitWithMode(
     const NvSciSyncFence* syncFence,
     NvSciSyncCpuWaitContext context,
     int64_t timeoutUs,
     NvSciSyncWaitMode waitMode);

 #if defined(__cplusplus)
 }
 #endif // __cplusplus

 #endif // INCLUDED_NVSCISYNC_H
