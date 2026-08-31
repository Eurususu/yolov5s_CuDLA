 /*
  * Copyright (c) 2018-2023, NVIDIA CORPORATION.  All rights reserved.
  *
  * NVIDIA CORPORATION and its licensors retain all intellectual property
  * and proprietary rights in and to this software, related documentation
  * and any modifications thereto.  Any use, reproduction, disclosure or
  * distribution of this software and related documentation without an express
  * license agreement from NVIDIA CORPORATION is strictly prohibited.
  */

 #ifndef INCLUDED_NVSCIIPC_H
 #define INCLUDED_NVSCIIPC_H

 #ifdef __cplusplus
 extern "C" {
 #endif

 #include <stdint.h>
 #include <stddef.h>
 #include "nvscierror.h"
 #include "nvscievent.h"

 /* use constant global version variable instead of macro for consistency with
  * version check API of existing NvSci family
  */

 static const uint32_t NvSciIpcMajorVersion = 1U;

 static const uint32_t NvSciIpcMinorVersion = 4U;

 /*******************************************************************/
 /************************ DATA TYPES *******************************/
 /*******************************************************************/

 typedef uint64_t NvSciIpcEndpoint;

 typedef struct NvSciIpcEndpointInfo NvSciIpcEndpointInfo;

 struct NvSciIpcEndpointInfo {
     uint32_t nframes;
     uint32_t frame_size;
 };

 #define NVSCIIPC_MAX_ENDPOINT_NAME   64U

 /* NvSciIPC Event type */
 #define NV_SCI_IPC_EVENT_READ           0x01U

 #define NV_SCI_IPC_EVENT_WRITE          0x02U

 #define NV_SCI_IPC_EVENT_CONN_EST       0x04U

 #define NV_SCI_IPC_EVENT_CONN_RESET     0x08U

 #define NV_SCI_IPC_EVENT_WRITE_EMPTY    0x10U

 #define NV_SCI_IPC_EVENT_ASYNC_ERROR    0x20U

 #define NV_SCI_IPC_EVENT_CONN_EST_ALL (NV_SCI_IPC_EVENT_CONN_EST | \
     NV_SCI_IPC_EVENT_WRITE | NV_SCI_IPC_EVENT_WRITE_EMPTY | \
     NV_SCI_IPC_EVENT_READ)

 #define NVSCIIPC_INFINITE_WAIT -1LL

 /* NvSciIpc Asynchronous erros */
 #define NV_SCI_ASYNC_PCIE_EDMA_XFER_ERROR            0x1U

 #define NV_SCI_ASYNC_PCIE_AER_UNCORRECTABLE_FATAL    0x2U

 #define NV_SCI_ASYNC_PCIE_AER_UNCORRECTABLE_NONFATAL 0x4U

 #define NV_SCI_ASYNC_PCIE_VALIDATION_ERROR           0x8U

 /*******************************************************************/
 /********************* FUNCTION TYPES ******************************/
 /*******************************************************************/

 NvSciError NvSciIpcInit(void);

 void NvSciIpcDeinit(void);

 NvSciError NvSciIpcOpenEndpoint(const char *endpoint, NvSciIpcEndpoint *handle);

 NvSciError NvSciIpcOpenEndpointWithEventService(const char *endpoint,
     NvSciIpcEndpoint *handle, NvSciEventService *eventService);

 NvSciError NvSciIpcGetEventNotifier(NvSciIpcEndpoint handle,
                NvSciEventNotifier **eventNotifier);

 void NvSciIpcCloseEndpoint(NvSciIpcEndpoint handle);

 NvSciError NvSciIpcCloseEndpointSafe(NvSciIpcEndpoint handle, bool clear);

 void NvSciIpcResetEndpoint(NvSciIpcEndpoint handle);

 NvSciError NvSciIpcResetEndpointSafe(NvSciIpcEndpoint handle);

 NvSciError NvSciIpcRead(NvSciIpcEndpoint handle, void *buf, size_t size,
     int32_t *bytes);

 NvSciError NvSciIpcReadSafe(NvSciIpcEndpoint handle, void *buf, uint32_t size,
     uint32_t *bytes);

 NvSciError NvSciIpcReadGetNextFrame(NvSciIpcEndpoint handle,
     const volatile void **buf);

 NvSciError NvSciIpcReadAdvance(NvSciIpcEndpoint handle);

 NvSciError NvSciIpcWrite(NvSciIpcEndpoint handle, const void *buf, size_t size,
     int32_t *bytes);

 NvSciError NvSciIpcWriteSafe(NvSciIpcEndpoint handle, const void *buf,
     uint32_t size, uint32_t *bytes);

 NvSciError NvSciIpcWriteGetNextFrame(NvSciIpcEndpoint handle,
     volatile void **buf);

 NvSciError NvSciIpcWriteAdvance(NvSciIpcEndpoint handle);

 NvSciError NvSciIpcGetEndpointInfo(NvSciIpcEndpoint handle,
                 NvSciIpcEndpointInfo *info);

 #ifndef __QNX__

 NvSciError NvSciIpcGetLinuxEventFd(NvSciIpcEndpoint handle, int32_t *fd);
 #endif /* !__QNX__ */

 NvSciError NvSciIpcGetEvent(NvSciIpcEndpoint handle, uint32_t *events);

 NvSciError NvSciIpcGetEventSafe(NvSciIpcEndpoint handle, uint32_t *events);

 #if defined(__QNX__)

 int32_t NvSciIpcWaitEventQnx(int chid, int64_t microseconds, uint32_t bytes,
     void *pulse);

 NvSciError NvSciIpcSetQnxPulseParam(NvSciIpcEndpoint handle,
     int32_t coid, int16_t pulsePriority, int16_t pulseCode,
     void *pulseValue);

 NvSciError NvSciIpcSetQnxPulseParamSafe(NvSciIpcEndpoint handle,
     int32_t coid, int16_t pulsePriority, int16_t pulseCode);


 NvSciError NvSciIpcInspectEventQnx(int32_t chid, uint16_t numEvents,
     uint32_t epCount, NvSciIpcEndpoint **epHandleArray);
 #endif /* __QNX__ */

 NvSciError NvSciIpcGetAsyncErrors(NvSciIpcEndpoint handle,  uint32_t* errors);

 NvSciError NvSciIpcEnableNotification(NvSciIpcEndpoint handle, bool flag);

 NvSciError NvSciIpcCheckVersionCompatibility(
     uint32_t majorVer,
     uint32_t minorVer,
     bool* isCompatible);

 #ifdef __cplusplus
 }
 #endif
 #endif /* INCLUDED_NVSCIIPC_H */
