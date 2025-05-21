# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: for zero-shot grafting, we would like to treat the image tokens as natural language tokens,
# so we use those template prompts to wrap the image tokens
# to make them more natural for the full-size language models to process
template_for_image_tokens = [
    "Based on the visual elements captured in this image of",
    "Drawing from what's shown in this photograph/illustration of",
    "Examining the details visible in this visual representation of",
    "Looking at the scene depicted in this image featuring",
    "Observing the components displayed in this visual of",
    "With attention to the specific details portrayed in this image of",
    "Considering the visual narrative presented in this scene of",
    "Noting the key elements highlighted in this visual capture of",
    "Reviewing the visual information documented in this image of",
    "Analyzing the pictorial evidence shown in this representation of",
    "In the context of what's depicted in this visual of",
    "Given the scene captured in this image showing",
    "Within the framework of what's illustrated here about",
    "From the perspective offered by this visual portrayal of",
    "Through the lens of what's represented in this image of",
    "Interpreting the visual cues present in this depiction of",
    "Decoding the visual information conveyed in this image of",
    "Making sense of the visual elements arranged in this scene of",
    "Processing the visual data presented in this capture of",
    "Reading the visual story told through this image of",
    "Focusing on the specific visual components shown in this image of",
    "Zeroing in on the particular elements depicted in this visual of",
    "Concentrating on the detailed representation provided in this image of",
    "Examining the precise visual information contained in this portrayal of",
    "Studying the exact visual features presented in this illustration of",
    "Let's explore what this image reveals about",
    "Let's break down the visual elements shown here regarding",
    "Let's analyze what we can see in this depiction of",
    "Let's examine the visual information captured about",
    "Let's investigate what this image tells us about",
]
