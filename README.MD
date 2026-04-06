# Numenor Math

---

## Numerical solvers and math utilities for Java

> **AUTHOR:** Piotr Jażdżyk <br>
> **LINKEDIN:** https://www.linkedin.com/in/pjazdzyk <br>

[![Numenor Math](https://img.shields.io/github/v/release/pjazdzyk/numenor-math?label=mumenor-math&color=13ADF3&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMi41bW0iIGhlaWdodD0iMTQuNW1tIiB2aWV3Qm94PSIwIDAgMjI1MCAxNDUwIj4NCiAgPHBvbHlnb24gZmlsbD0iIzUwN0QxNCIgcG9pbnRzPSIyMjQxLjAzLDE1Ljg4IDExMzYuMzgsMTUuODQgOTA1Ljg4LDQxNS4xIDIwMTAuNTMsNDE1LjA5IiAvPg0KICA8cG9seWdvbiBmaWxsPSIjNzFBQjIzIiBwb2ludHM9IjExMTYuMzgsMTUuODQgNjU1Ljk5LDE1Ljg0IDQ5NC4xNSwyOTYuMTcgNzI0LjM1LDY5NC44OCIgLz4NCiAgPHBvbHlnb24gZmlsbD0iIzhBQzkzNCIgcG9pbnRzPSI0ODQuMTUsMzA2LjE3IDI1NS4wNiw3MDIuOTYgMzg3LjY2LDkzMi42NCA4NDUuODMsOTMyLjYzIiAvPg0KICA8cG9seWdvbiBmaWxsPSIjNThEMEZGIiBwb2ludHM9Ii03LjE3LDE0NDAuMDkgMTA5Ny45NywxNDQwLjA4IDEzMjguNDcsMTA0MC44MyAyMjMuMzIsMTA0MC44NSIgLz4NCiAgPHBvbHlnb24gZmlsbD0iIzEzQURGMyIgcG9pbnRzPSIxNzM5LjA0LDExNjAuOTEgMTUwOS4wOSw3NjIuNjQgMTExNy45NywxNDQwLjA4IDExODYuOTMsMTQ0MC4wOCAxNTc3Ljg3LDE0NDAuMDgiIC8+DQogIDxwb2x5Z29uIGZpbGw9IiMwMzkzRDAiIHBvaW50cz0iMTk3OC44LDc1Mi45NiAxODQ2LjIsNTIzLjMgMTM4Ni42OCw1MjMuMyAxNzQ5LjA0LDExNTAuOTEiIC8+DQo8L3N2Zz4=)](https://github.com/pjazdzyk/numenor-math)
[![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=pjazdzyk_unitility&metric=vulnerabilities)](https://sonarcloud.io/summary/new_code?id=pjazdzyk_brent-dekker-solver) &nbsp;
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=pjazdzyk_unitility&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=pjazdzyk_brent-dekker-solver) &nbsp;
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=pjazdzyk_unitility&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=pjazdzyk_brent-dekker-solver) &nbsp;

---

## Table of Contents

- [Overview](#overview)
- [Available Solvers](#available-solvers)
  - [Brent-Dekker Solver](#brent-dekker-solver)
  - [Newton-Raphson Solver](#newton-raphson-solver)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Brent-Dekker Solver Guide](#brent-dekker-solver-guide)
- [Newton-Raphson Solver Guide](#newton-raphson-solver-guide)
- [NaN/Infinity Handling](#naninfinity-handling)
- [Tech Stack](#tech-stack)
- [License](#license)
- [Reference](#reference)

---

## Overview

**Numenor Math** is a Java library that groups numerical solvers and mathematical utilities. It provides robust, well-tested implementations of root-finding algorithms for single-variable equations.

The library currently includes two solvers:

| Solver | Type | Requires Bracket | Requires Initial Guess |
|--------|------|------------------|----------------------|
| **Brent-Dekker** | Bracketing (Bisection + Secant + Inverse Quadratic) | Yes (two points) | No |
| **Newton-Raphson** | Open (Derivative-based) | No | Yes (single point) |

---

## Available Solvers

### Brent-Dekker Solver

A Java implementation of the improved Brent-Dekker algorithm based on modifications proposed by Zhengqiu Zhang. Uses a combination of Secant and Bisection numerical schemes with inverse quadratic interpolation. Features a two-phase automatic counterpart points evaluation procedure.

**Best for:** Functions where you can provide a bracket (two points with opposite signs), robust convergence guaranteed.

### Newton-Raphson Solver

A derivative-based root-finding method that uses numerical differentiation for functions where the derivative is not available analytically. Converges quadratically near simple roots when the initial guess is sufficiently close.

**Best for:** Functions where you have a good initial guess and need fast convergence.

---

## Installation

Add Maven dependency to your `pom.xml`:

```xml
<dependency>
  <groupId>com.synerset</groupId>
  <artifactId>numenor-math</artifactId>
  <version>1.0.0</version>
</dependency>
```

---

## Quick Start

### Using Brent-Dekker Solver (bracketing method)

```java
import com.synerset.numenormath.solvers.BrentSolver;
import java.util.function.DoubleUnaryOperator;

// Create solver instance
BrentSolver solver = BrentSolver.of("MySolver");

// Define function: 2x + 10 = 0
DoubleUnaryOperator linear = x -> 2 * x + 10;

// Find root with bracket [−50, 50]
double root = solver.findRoot(linear, -50, 50);
// Result: -5.0
```

### Using Newton-Raphson Solver (open method)

```java
import com.synerset.numenormath.solvers.NewtonRaphsonSolver;
import java.util.function.DoubleUnaryOperator;

// Create solver instance
NewtonRaphsonSolver solver = NewtonRaphsonSolver.of("MySolver");

// Define function: x² - 4 = 0
DoubleUnaryOperator quadratic = x -> x * x - 4;

// Find root with initial guess
double root = solver.findRoot(quadratic, 3);
// Result: 2.0
```

### Static one-liner

```java
// Brent-Dekker
double root1 = BrentSolver.findRootOf(x -> 2 * x + 10, -10, 10);

// Newton-Raphson
double root2 = NewtonRaphsonSolver.findRootOf(x -> x * x - 25, 6);
```

---

## Brent-Dekker Solver Guide

### Creating a Solver

```java
// Default instance
BrentSolver solver = BrentSolver.of();

// Named instance (useful for log identification)
BrentSolver solver = BrentSolver.of("MySolver");
```

### Defining Functions

Any single-variable equation can be provided as a `DoubleUnaryOperator`:

**I. Linear function:** 2x + 10 = 0
```java
DoubleUnaryOperator linear = x -> 2 * x + 10;
```

**II. Quadratic function:** 2x² + 5x - 3 = 0
```java
DoubleUnaryOperator quadratic = x -> 2 * x * x + 5 * x - 3;
```

**III. Complex nested function:**
```java
DoubleUnaryOperator logNested = x -> 93.3519196629417 - 
    (-237300 * Math.log(0.001638 * x) / (1000 * Math.log(0.001638 * x) - 17269));
```

### Finding Roots

The solver requires two counterpart points `a` and `b` where f(a) and f(b) have opposite signs. If invalid points are provided, the two-phase evaluation procedure attempts to find valid brackets automatically.

```java
// With function and bracket
double root = solver.findRoot(linear, -50, 50);

// Using default counterpart points (+50/-50)
double root = solver.findRoot(linear);

// Adjusting counterpart points for different roots
solver.setCounterpartPoints(-1, 2);
double root = solver.findRoot(quadratic);
```

### Configuration

```java
solver.setAccuracy(1E-12);           // Set convergence tolerance
solver.setIterationsLimit(200);      // Maximum iterations
solver.showDebugLogs(true);          // Enable diagnostic output
solver.showSummaryLogs(true);        // Enable summary logs
```

---

## Newton-Raphson Solver Guide

### Creating a Solver

```java
// Default instance
NewtonRaphsonSolver solver = NewtonRaphsonSolver.of();

// Named instance
NewtonRaphsonSolver solver = NewtonRaphsonSolver.of("MySolver");
```

### Finding Roots

The Newton-Raphson method requires only a single initial guess:

```java
// Define function: x² - 4 = 0
DoubleUnaryOperator func = x -> x * x - 4;

// Find root starting from x = 3
double root = solver.findRoot(func, 3);
// Result: 2.0

// Find the other root with different initial guess
double root2 = solver.findRoot(func, -3);
// Result: -2.0
```

### Configuration

```java
solver.setAccuracy(1E-12);              // Set convergence tolerance
solver.setIterationsLimit(200);         // Maximum iterations
solver.setDerivativeStep(1E-8);         // Numerical differentiation step
solver.setInitialGuess(5);              // Set initial guess separately
solver.showDebugLogs(true);             // Enable diagnostic output
solver.showSummaryLogs(true);           // Enable summary logs
```

### When to Use Newton-Raphson

| Advantage | Limitation |
|-----------|------------|
| Faster convergence near simple roots | Requires good initial guess |
| No bracket needed | May diverge for poor guesses |
| Works with single point | Struggles with zero derivatives |
| Quadratic convergence | Can cycle for certain functions |

---

## NaN/Infinity Handling

Both solvers throw `BrentSolverException` by default when NaN or Infinity values are detected. You can switch to graceful mode:

### Brent-Dekker

```java
solver.setFailForNaN(false);
double result = solver.findRoot(someFunction, a, b);
// Returns best approximation found so far instead of throwing
```

### Newton-Raphson

```java
solver.setFailForNaN(false);
double result = solver.findRoot(someFunction, initialGuess);
// Stops gracefully and returns current approximation
```

---

## What's New in 1.0.0

- **Two-phase counterpart evaluation:** Added exponential bracket expansion as fallback for Brent-Dekker solver
- **Newton-Raphson solver:** New derivative-based root-finding method with numerical differentiation
- **Near-zero counterpart safeguard:** Fixed degenerate results when initial point is close to zero
- **Reduced function evaluations:** Eliminated redundant function calls in iteration loops
- **Expanded test coverage:** Comprehensive tests for standard functions, badly behaved functions, engineering equations, and edge cases

---

## Tech Stack

**Core:** <br>
![Java 21](https://img.shields.io/badge/Java%2021-orange?style=for-the-badge) &nbsp;
![Maven](https://img.shields.io/badge/Apache%20Maven-C71A36?style=for-the-badge&logo=apachemaven&logoColor=white) &nbsp;

**Testing:** <br>
![JUnit 5](https://img.shields.io/badge/JUnit%205-25A162?style=for-the-badge&logo=junit5&logoColor=white) &nbsp;

**CI/CD:** <br>
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white) &nbsp;
![SonarCloud](https://img.shields.io/badge/Sonar%20Cloud-F3702A?style=for-the-badge&logo=sonarcloud&logoColor=white) &nbsp;

---

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

## Reference

- [1] BRENT-DEKKER ITERATIVE SOLVER - Modified algorithm proposed by Zhengqiu Zhang / International Journal of Experimental Algorithms (IJEA), Volume (2) : Issue (1) : 2011
- [2] Press, W.H., Teukolsky, S.A., Vetterling, W.T. & Flannery, B.P. (2007). *Numerical Recipes: The Art of Scientific Computing*, 3rd Edition, Cambridge University Press. — Section 9.4: Newton-Raphson Method

---

## Acknowledgments

I extend my heartfelt gratitude to the [Silesian University of Technology](https://www.polsl.pl/en/) as it is here 
that my professional identity was forged, transforming me into an engineer to my very core.
Amicus Plato, sed magis amica veritas.