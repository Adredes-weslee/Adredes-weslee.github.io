---
layout: post
title: "Building a Learning Personalization Engine from Platform Evidence"
date: 2026-04-19 09:00:00 +0800
categories: [ai-platforms, edtech, personalization]
tags: [knowledge-tracing, personalization, learning-analytics, recommendations, pytest, playwright]
author: Wes Lee
feature_image: /assets/images/article-heroes/2026-04-19-building-learning-personalization-engine.jpg
project_page: /projects/learning-personalization-engine/
display_title: "Learning Personalization from Platform Evidence"
archive_title: "Learning Personalization Engine"
---

## Introduction: personalization starts with evidence quality

Learning personalization is easy to describe and difficult to build responsibly. A useful system needs more than a learner score or a recommendation list. It needs to know where the evidence came from, how platform events were normalized, which assumptions were applied, and how a teacher or operator can review the output.

This project is a private-code case study for that problem. The public version intentionally omits employer and deployment-specific details. The relevant engineering story is the architecture: authenticated probes, event contracts, mastery tracing, recommendation inventory, and teacher-facing review surfaces.

> Related: for the shorter case-study version, see the [Learning Personalization Engine project page](/projects/learning-personalization-engine/).

## The product boundary is platform evidence, not a generic learner model

The repo treats platform observations as raw evidence that must be cleaned before they can support personalization. That is the right boundary. Without an event contract, the model can only guess what an activity means.

The useful parts of the system are therefore upstream of the recommendation:

- event parsing and metadata normalization
- profile rollups for learner activity
- mastery updates that preserve assumptions
- recommendation inventory rules
- teacher-view summaries that expose what the system believes and why

This makes the system less flashy than a generic AI tutor demo, but more usable as an operations layer.

## Knowledge tracing is only one layer

Bayesian-style mastery tracing gives the engine a structured way to update learner state, but it is not the whole product. The stronger design choice is separating the tracing layer from the recommendation and review layers.

That split matters because a mastery estimate should not automatically become an intervention. The system still needs to ask what recommendation is appropriate, what evidence supports it, and whether the output should be reviewed before it reaches a learner-facing workflow.

## Verification keeps the private system inspectable

Because the underlying platform surface is private, the public portfolio should not depend on raw screenshots or deployment-specific claims. The repo handles that by keeping fixture-driven validation and tests close to the core logic.

The local test suite covers event normalization, knowledge tracing, metadata parsing, recommendation inventory behavior, local stores, pilot configuration, and end-to-end output generation. That is the public signal that matters: the system is not just a narrative around a private platform. It has executable checks around the parts that make personalization defensible.

## The broader lesson: recommendations need review surfaces

Personalization work becomes more credible when it is framed as decision support rather than hidden automation. The useful interface is not only "what should happen next?" It is also:

- what evidence led to this recommendation
- what mastery state changed
- which assumptions are still uncertain
- what a teacher or operator should review before acting

That is the main portfolio takeaway. The system uses model logic, but the product value comes from turning platform evidence into inspectable recommendations.
