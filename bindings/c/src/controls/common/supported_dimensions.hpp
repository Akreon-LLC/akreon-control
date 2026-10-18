/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Supported runtime dimensions compiled into the control bindings.
using StateFeedbackStateDims = SupportedDims<4>;
using StateFeedbackInputDims = SupportedDims<1>;

using StateObserverStateDims = SupportedDims<4>;
using StateObserverMeasurementDims = SupportedDims<2>;

using H2StateFeedbackStateDims = SupportedDims<4>;
using H2StateFeedbackExogenousDims = SupportedDims<1>;
using H2StateFeedbackControlDims = SupportedDims<1>;
using H2StateFeedbackPerformanceDims = SupportedDims<5>;

using HinfStateFeedbackStateDims = SupportedDims<4, 7, 9>;
using HinfStateFeedbackExogenousDims = SupportedDims<1, 2>;
using HinfStateFeedbackControlDims = SupportedDims<1>;
using HinfStateFeedbackPerformanceDims = SupportedDims<4, 5>;

using H2StateObserverStateDims = SupportedDims<4>;
using H2StateObserverExogenousDims = SupportedDims<3>;
using H2StateObserverPerformanceDims = SupportedDims<4>;
using H2StateObserverMeasurementDims = SupportedDims<2>;

using HinfStateObserverStateDims = SupportedDims<4>;
using HinfStateObserverExogenousDims = SupportedDims<3>;
using HinfStateObserverPerformanceDims = SupportedDims<4>;
using HinfStateObserverMeasurementDims = SupportedDims<2>;

using PolytopicStateFeedbackVertexDims = SupportedDims<2>;
using PolytopicStateFeedbackStateDims = SupportedDims<4>;
using PolytopicStateFeedbackInputDims = SupportedDims<1>;

using PolytopicStateObserverVertexDims = SupportedDims<2>;
using PolytopicStateObserverStateDims = SupportedDims<4>;
using PolytopicStateObserverMeasurementDims = SupportedDims<2>;

using H2PolytopicStateFeedbackVertexDims = SupportedDims<2>;
using H2PolytopicStateFeedbackStateDims = SupportedDims<4>;
using H2PolytopicStateFeedbackExogenousDims = SupportedDims<1>;
using H2PolytopicStateFeedbackControlDims = SupportedDims<1>;
using H2PolytopicStateFeedbackPerformanceDims = SupportedDims<5>;

using HinfPolytopicStateFeedbackVertexDims = SupportedDims<2>;
using HinfPolytopicStateFeedbackStateDims = SupportedDims<4>;
using HinfPolytopicStateFeedbackExogenousDims = SupportedDims<1>;
using HinfPolytopicStateFeedbackControlDims = SupportedDims<1>;
using HinfPolytopicStateFeedbackPerformanceDims = SupportedDims<5>;

using H2PolytopicStateObserverVertexDims = SupportedDims<2>;
using H2PolytopicStateObserverStateDims = SupportedDims<4>;
using H2PolytopicStateObserverExogenousDims = SupportedDims<3>;
using H2PolytopicStateObserverPerformanceDims = SupportedDims<4>;
using H2PolytopicStateObserverMeasurementDims = SupportedDims<2>;

using HinfPolytopicStateObserverVertexDims = SupportedDims<2>;
using HinfPolytopicStateObserverStateDims = SupportedDims<4>;
using HinfPolytopicStateObserverExogenousDims = SupportedDims<3>;
using HinfPolytopicStateObserverPerformanceDims = SupportedDims<4>;
using HinfPolytopicStateObserverMeasurementDims = SupportedDims<2>;

// Unions used by shared C ABI validation paths.
using SupportedStateDims = SupportedDims<4, 7, 9>;
using SupportedSignalDims = SupportedDims<1, 2>;
using SupportedGeneralizedPortDims = SupportedDims<1, 2, 3, 4, 5>;
using SupportedPolytopeDims = SupportedDims<2>;

// Exact tuples avoid compiling the Cartesian product for augmented Hinf plants.
using HinfFeedbackShapes =
    ShapeList<SupportedShape<4, 1, 1, 5>, SupportedShape<4, 1, 1, 4>,
              SupportedShape<7, 1, 1, 4>, SupportedShape<9, 2, 1, 5>>;
