 /*
  * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
  *
  * NVIDIA CORPORATION and its licensors retain all intellectual property
  * and proprietary rights in and to this software, related documentation
  * and any modifications thereto.  Any use, reproduction, disclosure or
  * distribution of this software and related documentation without an express
  * license agreement from NVIDIA CORPORATION is strictly prohibited.
  */

 #ifndef INCLUDED_NVSCIEVENT_H
 #define INCLUDED_NVSCIEVENT_H

 #ifdef __cplusplus
 extern "C" {
 #endif

 #include <string.h>
 #include <stdint.h>
 #include <stdlib.h>
 #include <stdbool.h>
 #include "nvscierror.h"

 /* use constant global version variable instead of macro for consistency with
  * version check API of existing NvSci family
  */

 static const uint32_t NvSciEventMajorVersion = 1U;

 static const uint32_t NvSciEventMinorVersion = 4U;

 /*****************************************************************************/
 /*                               DATA TYPES                                  */
 /*****************************************************************************/

 #define NV_SCI_EVENT_INFINITE_WAIT -1
 #define NV_SCI_EVENT_PRIORITIES 4

 typedef struct NvSciEventService NvSciEventService;
 typedef struct NvSciEventNotifier NvSciEventNotifier;
 typedef struct NvSciEventLoopService NvSciEventLoopService;
 typedef struct NvSciNativeEvent NvSciNativeEvent;
 typedef struct NvSciLocalEvent NvSciLocalEvent;

 typedef struct NvSciTimerEvent NvSciTimerEvent;
 typedef struct NvSciEventLoop NvSciEventLoop;

 struct NvSciLocalEvent {
     NvSciEventNotifier* eventNotifier;

     NvSciError (*Signal)(NvSciLocalEvent* thisLocalEvent);

     void (*Delete)(NvSciLocalEvent* thisLocalEvent);
 };

 struct NvSciEventService {
     NvSciError (*CreateNativeEventNotifier)(
             NvSciEventService* thisEventService,
             NvSciNativeEvent* nativeEvent,
             NvSciEventNotifier** newEventNotifier);

     NvSciError (*CreateLocalEvent)(
             NvSciEventService* thisEventService,
             NvSciLocalEvent** newLocalEvent);

     NvSciError (*CreateTimerEvent)(
             NvSciEventService* thisEventService,
             NvSciTimerEvent** newTimerEvent);

     void (*Delete)(NvSciEventService* thisEventService);
 };

 struct NvSciEventNotifier {
     NvSciError (*SetHandler)(NvSciEventNotifier* thisEventNotifier,
             void (*callback)(void* cookie),
             void* cookie,
             uint32_t priority);

     void (*Delete)(NvSciEventNotifier* thisEventNotifier);
 };

 NvSciError NvSciEventLoopServiceCreate(
         size_t maxEventLoops,
         NvSciEventLoopService** newEventLoopService);

 NvSciError NvSciEventLoopServiceCreateSafe(
         size_t maxEventLoops,
         void* config,
         NvSciEventLoopService** newEventLoopService);

 NvSciError NvSciEventLoopServiceCreateSafeX(
         size_t maxEventLoops,
         void* config,
         NvSciEventLoopService** newEventLoopService);

 struct NvSciEventLoopService {
     NvSciEventService EventService;

     NvSciError (*CreateEventLoop)(NvSciEventLoopService* eventLoopService,
             NvSciEventLoop** eventLoop);

     NvSciError (*WaitForEvent)(
             NvSciEventNotifier* eventNotifier,
             int64_t microseconds);

     NvSciError (*WaitForMultipleEvents)(
             NvSciEventNotifier* const * eventNotifierArray,
             size_t eventNotifierCount,
             int64_t microseconds,
             bool* newEventArray);

     NvSciError (*WaitForMultipleEventsExt)(
             NvSciEventService *eventService,
             NvSciEventNotifier* const * eventNotifierArray,
             size_t eventNotifierCount,
             int64_t microseconds,
             bool* newEventArray);
 };

 #ifdef __QNX__

 NvSciError NvSciEventInspect(
         NvSciEventService *thisEventService,
         uint32_t numEvents,
         uint32_t eventNotifierCount,
         NvSciEventNotifier** eventNotifierArray);
 #endif /* __QNX__ */

 NvSciError NvSciEventMoveNotifier(
         NvSciEventService *oldEventService,
         NvSciEventService *newEventService,
         NvSciEventNotifier *eventNotifier);

 NvSciError NvSciEventCheckVersionCompatibility(
     uint32_t majorVer,
     uint32_t minorVer,
     bool* isCompatible);

 #ifdef __cplusplus
 }
 #endif
 #endif /* INCLUDED_NVSCIEVENT_H */
