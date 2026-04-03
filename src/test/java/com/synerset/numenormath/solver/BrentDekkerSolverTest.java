package com.synerset.numenormath.solver;

import com.synerset.numenormath.exception.NumenorSolverException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.function.DoubleUnaryOperator;
import java.util.stream.Stream;

class BrentDekkerSolverTest {

    private BrentDekkerSolver solver;

    @BeforeEach
    void setUp() {
        solver = BrentDekkerSolver.of("Test-SOLVER");
        solver.toggleDebugLogs(true);
    }

    // Original tests

    @Test
    @DisplayName("should return root for simple linear function")
    void findRoot_givenSingleVariableFunction_returnsRoot() {
        // Arrange
        DoubleUnaryOperator func = p -> (p + 10) / 20;
        var expectedRoot = -10;

        // Act
        var actualRoot = solver.findRoot(func);

        // Assert
        Assertions.assertEquals(expectedRoot, actualRoot, 1E-10);
    }

    @Test
    @DisplayName("should return one of two roots within specified solution boundary")
    void findRoot_givenQuadraticFunction_returnRoot() {
        // Arrange
        DoubleUnaryOperator quadraticFunction = x -> 2 * x * x + 5 * x - 3;
        var expectedFirstRoot = -3;
        var expectedSecondRoot = 0.5;

        // Act
        var actualFirstRoot = solver.findRoot(quadraticFunction);
        solver.setCounterpartPoints(-1, 2);
        var actualSecondRoot = solver.findRoot(quadraticFunction);

        // Assert
        Assertions.assertEquals(expectedFirstRoot, actualFirstRoot, 1E-10);
        Assertions.assertEquals(expectedSecondRoot, actualSecondRoot, 1E-10);
    }

    @ParameterizedTest
    @MethodSource("polyTestInlineData")
    @DisplayName("should return root for nested log function for series of counterpart points which brakes brent-decker counterpart points condition")
    void findRoot_givenPolynomialFunction_returnRoot(double pointA, double pointB) {
        // Arrange
        DoubleUnaryOperator func = p -> 93.3519196629417 - (-237300 * Math.log(0.001638 * p) / (1000 * Math.log(0.001638 * p) - 17269));
        var expectedRoot = 80000;

        //Act
        var actualRoot = solver.findRoot(func, pointA, pointB);

        //Assert
        Assertions.assertEquals(actualRoot, expectedRoot, 1E-9);
    }

    static Stream<Arguments> polyTestInlineData() {
        return Stream.of(
                Arguments.of(50000, 120000),
                Arguments.of(80000, 200000),
                Arguments.of(80000, 80000),
                Arguments.of(20000, 80000),
                Arguments.of(10000, 20000)
        );
    }

    @Test
    @DisplayName("should throw an exception if point evaluation procedure fails to determine valid counterpart points")
    void findRoot_givenAcosFunction_throwsSolverResultException() {
        // Arrange
        solver.setCounterpartPoints(10, 5);
        DoubleUnaryOperator func = x -> Math.acos(x / 2);

        // Assert
        Assertions.assertThrows(NumenorSolverException.class, () -> solver.findRoot(func));
    }

    // Standard function tests

    @Nested
    @DisplayName("Standard mathematical functions")
    class StandardFunctionTests {

        @Test
        @DisplayName("should find root of identity function f(x) = x at zero")
        void findRoot_identityFunction() {
            DoubleUnaryOperator func = x -> x;
            double root = solver.findRoot(func, -5, 5);
            Assertions.assertEquals(0.0, root, 1E-10);
        }

        @Test
        @DisplayName("should find root of cubic function x^3 - 2x - 5")
        void findRoot_cubicFunction() {
            DoubleUnaryOperator func = x -> x * x * x - 2 * x - 5;
            double root = solver.findRoot(func, 1, 3);
            // Verify by substitution
            Assertions.assertEquals(0.0, func.applyAsDouble(root), 1E-10);
        }

        @Test
        @DisplayName("should find root of sin(x) near pi")
        void findRoot_sinNearPi() {
            DoubleUnaryOperator func = Math::sin;
            double root = solver.findRoot(func, 2.5, 3.8);
            Assertions.assertEquals(Math.PI, root, 1E-10);
        }

        @Test
        @DisplayName("should find root of cos(x) near pi/2")
        void findRoot_cosNearPiOver2() {
            DoubleUnaryOperator func = Math::cos;
            double root = solver.findRoot(func, 1, 2);
            Assertions.assertEquals(Math.PI / 2, root, 1E-10);
        }

        @Test
        @DisplayName("should find root of exponential-linear mix: e^x - 3x")
        void findRoot_exponentialLinearMix() {
            DoubleUnaryOperator func = x -> Math.exp(x) - 3 * x;
            double root = solver.findRoot(func, 0, 1);
            Assertions.assertEquals(0.0, func.applyAsDouble(root), 1E-10);
        }

        @Test
        @DisplayName("should find root of tan(x) - 1 near pi/4")
        void findRoot_tanMinusOne() {
            DoubleUnaryOperator func = x -> Math.tan(x) - 1;
            double root = solver.findRoot(func, 0, 1);
            Assertions.assertEquals(Math.PI / 4, root, 1E-10);
        }

        @Test
        @DisplayName("should find root of ln(x) at x=1")
        void findRoot_naturalLog() {
            DoubleUnaryOperator func = Math::log;
            double root = solver.findRoot(func, 0.1, 5);
            Assertions.assertEquals(1.0, root, 1E-10);
        }

        @Test
        @DisplayName("should find root of x*e^x - 1 (Lambert W related)")
        void findRoot_lambertW() {
            DoubleUnaryOperator func = x -> x * Math.exp(x) - 1;
            double root = solver.findRoot(func, 0, 1);
            Assertions.assertEquals(0.0, func.applyAsDouble(root), 1E-10);
        }

        @Test
        @DisplayName("should find root of high-degree polynomial x^7 - 1")
        void findRoot_highDegreePolynomial() {
            DoubleUnaryOperator func = x -> Math.pow(x, 7) - 1;
            double root = solver.findRoot(func, 0, 2);
            Assertions.assertEquals(1.0, root, 1E-10);
        }
    }

    // Badly behaved function tests

    @Nested
    @DisplayName("Badly behaved and challenging functions")
    class BadlyBehavedFunctionTests {

        @Test
        @DisplayName("should find root of very flat function near root: x^3 * 1e-8")
        void findRoot_veryFlatFunction() {
            DoubleUnaryOperator func = x -> x * x * x * 1E-8;
            double root = solver.findRoot(func, -1, 1);
            Assertions.assertEquals(0.0, root, 1E-3);
        }

        @Test
        @DisplayName("should find root of very steep function: 1000*x - 500")
        void findRoot_steepLinearFunction() {
            DoubleUnaryOperator func = x -> 1000 * x - 500;
            double root = solver.findRoot(func, -10, 10);
            Assertions.assertEquals(0.5, root, 1E-10);
        }

        @Test
        @DisplayName("should find root of function with large derivative: e^(10x) - 1e5")
        void findRoot_rapidlyGrowingExponential() {
            DoubleUnaryOperator func = x -> Math.exp(10 * x) - 1E5;
            double root = solver.findRoot(func, 0, 2);
            double expected = Math.log(1E5) / 10;
            Assertions.assertEquals(expected, root, 1E-10);
        }

        @Test
        @DisplayName("should find root near singularity: 1/x - 2 (root at 0.5)")
        void findRoot_nearSingularity() {
            DoubleUnaryOperator func = x -> 1.0 / x - 2;
            double root = solver.findRoot(func, 0.01, 5);
            Assertions.assertEquals(0.5, root, 1E-10);
        }

        @Test
        @DisplayName("should find root of oscillating function: sin(10x) near first positive root")
        void findRoot_oscillatingFunction() {
            DoubleUnaryOperator func = x -> Math.sin(10 * x);
            double root = solver.findRoot(func, 0.1, 0.5);
            Assertions.assertEquals(Math.PI / 10, root, 1E-10);
        }

        @Test
        @DisplayName("should find root of function with nearly zero derivative at root: (x-1)^3")
        void findRoot_cubicWithMultipleRoot() {
            // (x-1)^3 has a triple root at x=1 — derivative is zero at the root
            DoubleUnaryOperator func = x -> (x - 1) * (x - 1) * (x - 1);
            double root = solver.findRoot(func, -1, 3);
            Assertions.assertEquals(1.0, root, 1E-3); // relaxed tolerance for multiple root
        }

        @Test
        @DisplayName("should find root of function with sharp corner behavior: |x| - 0.5 approximated smoothly")
        void findRoot_sharpCornerApproximation() {
            // sqrt(x^2 + epsilon) - 0.5 approximates |x| - 0.5 but is smooth
            DoubleUnaryOperator func = x -> Math.sqrt(x * x + 1E-12) - 0.5;
            double root = solver.findRoot(func, 0, 2);
            Assertions.assertEquals(0.5, root, 1E-5);
        }

        @Test
        @DisplayName("should find root of Kepler's equation for orbital mechanics")
        void findRoot_keplersEquation() {
            // M = E - e*sin(E), solve for E given M = 0.5, e = 0.9 (high eccentricity)
            double M = 0.5;
            double e = 0.9;
            DoubleUnaryOperator func = E -> E - e * Math.sin(E) - M;
            double root = solver.findRoot(func, 0, Math.PI);
            Assertions.assertEquals(0.0, func.applyAsDouble(root), 1E-10);
        }

        @Test
        @DisplayName("should find root for function with very large root value")
        void findRoot_largeRootValue() {
            // x - 1e6
            DoubleUnaryOperator func = x -> x - 1E6;
            double root = solver.findRoot(func, 999000, 1001000);
            Assertions.assertEquals(1E6, root, 1E-5);
        }

        @Test
        @DisplayName("should find root for function with very small root value")
        void findRoot_smallRootValue() {
            // x - 1e-8
            DoubleUnaryOperator func = x -> x - 1E-8;
            double root = solver.findRoot(func, -1, 1);
            Assertions.assertEquals(1E-8, root, 1E-10);
        }
    }

    // Counterpart point evaluation tests

    @Nested
    @DisplayName("Counterpart point evaluation and bracket expansion")
    class CounterpartPointTests {

        @Test
        @DisplayName("should find root when both initial points are on the same side (positive)")
        void findRoot_bothPointsPositiveSide() {
            DoubleUnaryOperator func = x -> x - 5;
            double root = solver.findRoot(func, 10, 20);
            Assertions.assertEquals(5.0, root, 1E-10);
        }

        @Test
        @DisplayName("should find root when both initial points are on the same side (negative)")
        void findRoot_bothPointsNegativeSide() {
            DoubleUnaryOperator func = x -> x + 5;
            double root = solver.findRoot(func, -20, -10);
            Assertions.assertEquals(-5.0, root, 1E-10);
        }

        @Test
        @DisplayName("should find root when initial points are very close together")
        void findRoot_veryCloseInitialPoints() {
            DoubleUnaryOperator func = x -> x * x - 4;
            double root = solver.findRoot(func, 1.99, 2.01);
            Assertions.assertEquals(2.0, root, 1E-10);
        }

        @Test
        @DisplayName("should find root when initial interval is very wide")
        void findRoot_veryWideInitialInterval() {
            DoubleUnaryOperator func = x -> x - 42;
            double root = solver.findRoot(func, -10000, 10000);
            Assertions.assertEquals(42.0, root, 1E-10);
        }

        @Test
        @DisplayName("should find root when one counterpart point is at the root")
        void findRoot_pointAtRoot() {
            DoubleUnaryOperator func = x -> x * x - 9;
            double root = solver.findRoot(func, 3, 10);
            Assertions.assertEquals(3.0, root, 1E-10);
        }

        @Test
        @DisplayName("should find root when both counterpart points are equal")
        void findRoot_equalCounterpartPoints() {
            DoubleUnaryOperator func = x -> x - 7;
            double root = solver.findRoot(func, 10, 10);
            Assertions.assertEquals(7.0, root, 1E-9);
        }

        @Test
        @DisplayName("should find root with reversed counterpart point order (a > b)")
        void findRoot_reversedCounterpartOrder() {
            DoubleUnaryOperator func = x -> x * x - 2;
            double root = solver.findRoot(func, 3, 0);
            Assertions.assertEquals(Math.sqrt(2), root, 1E-10);
        }
    }

    // NaN and infinity handling tests

    @Nested
    @DisplayName("NaN and Infinity handling")
    class NaNInfinityTests {

        @Test
        @DisplayName("should throw exception when function produces NaN with failForNaN enabled")
        void findRoot_nanWithFailEnabled_throwsException() {
            solver.setFailForNaN(true);
            solver.setCounterpartPoints(-2, 2);
            DoubleUnaryOperator func = x -> Math.log(x); // NaN for negative x
            Assertions.assertThrows(NumenorSolverException.class, () -> solver.findRoot(func));
        }

        @Test
        @DisplayName("should not throw but stop when function produces NaN with failForNaN disabled")
        void findRoot_nanWithFailDisabled_returnsWithoutException() {
            solver.setFailForNaN(false);
            solver.setCounterpartPoints(-2, 2);
            DoubleUnaryOperator func = x -> Math.log(x);
            // Should not throw - instead stops gracefully
            Assertions.assertDoesNotThrow(() -> solver.findRoot(func));
        }

        @Test
        @DisplayName("should throw exception for function that produces infinity")
        void findRoot_infinityResult_throwsException() {
            solver.setFailForNaN(true);
            solver.setCounterpartPoints(-1, 1);
            // Function that directly produces infinity within the solving interval
            DoubleUnaryOperator func = x -> 1.0 / x;
            Assertions.assertThrows(NumenorSolverException.class, () -> solver.findRoot(func));
        }
    }

    // Solver configuration tests

    @Nested
    @DisplayName("Solver configuration and state management")
    class ConfigurationTests {

        @Test
        @DisplayName("should respect custom accuracy setting")
        void findRoot_customAccuracy() {
            solver.setAccuracy(1E-4);
            DoubleUnaryOperator func = x -> x * x - 2;
            double root = solver.findRoot(func, 0, 2);
            // Should be accurate to at least 1E-4
            Assertions.assertEquals(Math.sqrt(2), root, 1E-4);
        }

        @Test
        @DisplayName("should stop at iteration limit and return best approximation")
        void findRoot_iterationLimitReached() {
            solver.setIterationsLimit(3);
            DoubleUnaryOperator func = x -> x * x * x - x - 1;
            // With only 3 iterations, result should still be a reasonable approximation
            double root = solver.findRoot(func, 1, 2);
            Assertions.assertEquals(0.0, func.applyAsDouble(root), 0.1);
        }

        @Test
        @DisplayName("should be reusable for multiple solves after reset")
        void findRoot_solverReuse() {
            DoubleUnaryOperator func1 = x -> x - 3;
            double root1 = solver.findRoot(func1, 0, 10);
            Assertions.assertEquals(3.0, root1, 1E-10);

            DoubleUnaryOperator func2 = x -> x * x - 16;
            double root2 = solver.findRoot(func2, 0, 10);
            Assertions.assertEquals(4.0, root2, 1E-10);
        }

        @Test
        @DisplayName("should work with static factory findRootOf method")
        void findRootOf_staticFactory() {
            double root = BrentDekkerSolver.findRootOf(x -> x * x - 25, 0, 10);
            Assertions.assertEquals(5.0, root, 1E-10);
        }

        @Test
        @DisplayName("should throw exception for null function")
        void findRoot_nullFunction_throwsException() {
            Assertions.assertThrows(NumenorSolverException.class, () -> new BrentDekkerSolver("test", null, -1, 1));
        }

        @Test
        @DisplayName("should return correct iteration count after solve")
        void findRoot_iterationCountTracked() {
            DoubleUnaryOperator func = x -> x - 1;
            solver.findRoot(func, -10, 10);
            Assertions.assertTrue(solver.lastResultIterationCount() > 0, "Counter should be positive after solve");
            Assertions.assertTrue(solver.lastResultIterationCount() <= 100, "Counter should not exceed default iteration limit");
        }
    }

    // Engineering and real-world function tests

    @Nested
    @DisplayName("Real-world and engineering functions")
    class EngineeringFunctionTests {

        @Test
        @DisplayName("should solve Colebrook-White equation for pipe friction factor")
        void findRoot_colebrookWhite() {
            // Colebrook-White: 1/sqrt(f) = -2*log10(e/(3.7*D) + 2.51/(Re*sqrt(f)))
            // Rearranged: 1/sqrt(f) + 2*log10(e/(3.7*D) + 2.51/(Re*sqrt(f))) = 0
            double roughness = 0.0001; // e
            double diameter = 0.1;     // D
            double reynolds = 100000;  // Re

            DoubleUnaryOperator func = f -> {
                double sqrtF = Math.sqrt(Math.abs(f));
                if (sqrtF < 1E-15) return Double.MAX_VALUE;
                return 1.0 / sqrtF + 2.0 * Math.log10(roughness / (3.7 * diameter) + 2.51 / (reynolds * sqrtF));
            };

            double root = solver.findRoot(func, 0.005, 0.08);
            // Verify the result satisfies the equation (use original formula for verification)
            double sqrtRoot = Math.sqrt(root);
            double residual = 1.0 / sqrtRoot + 2.0 * Math.log10(roughness / (3.7 * diameter) + 2.51 / (reynolds * sqrtRoot));
            Assertions.assertEquals(0.0, residual, 1E-8);
            // Friction factor should be in realistic range
            Assertions.assertTrue(root > 0.005 && root < 0.08, "Friction factor should be in realistic range");
        }

        @Test
        @DisplayName("should solve Antoine equation for vapor pressure temperature")
        void findRoot_antoineEquation() {
            // Antoine equation for water (mmHg units): log10(P) = A - B/(C + T)
            // Solve for T given P = 760 mmHg (boiling point at atmospheric pressure)
            double A = 8.07131;
            double B = 1730.63;
            double C = 233.426;
            double targetP = 760.0; // mmHg

            DoubleUnaryOperator func = T -> Math.pow(10, A - B / (C + T)) - targetP;

            double root = solver.findRoot(func, 50, 150);
            Assertions.assertEquals(0.0, func.applyAsDouble(root), 1E-6);
            // Boiling point of water should be near 100°C
            Assertions.assertEquals(100.0, root, 1.0);
        }

        @Test
        @DisplayName("should solve ideal gas law for temperature: PV = nRT")
        void findRoot_idealGasLaw() {
            // Solve PV = nRT for T, given P=101325 Pa, V=0.0224 m^3, n=1 mol, R=8.314
            double P = 101325;
            double V = 0.0224;
            double n = 1.0;
            double R = 8.314;

            DoubleUnaryOperator func = T -> P * V - n * R * T;

            double root = solver.findRoot(func, 200, 400);
            double expected = P * V / (n * R);
            Assertions.assertEquals(expected, root, 1E-8);
        }
    }

    // Convergence and performance tests

    @Nested
    @DisplayName("Convergence behavior")
    class ConvergenceTests {

        @Test
        @DisplayName("should converge quickly for well-behaved quadratic")
        void findRoot_convergenceSpeed_quadratic() {
            DoubleUnaryOperator func = x -> x * x - 4;
            solver.findRoot(func, 0, 5);
            // Brent should converge in far fewer than 100 iterations for this simple case
            Assertions.assertTrue(solver.lastResultIterationCount() < 20,
                    "Should converge in fewer than 20 iterations, actual: " + solver.lastResultIterationCount());
        }

        @Test
        @DisplayName("should converge for Wilkinson-type polynomial near clustered roots")
        void findRoot_wilkinsonPolynomial() {
            // (x-1)(x-2)(x-3) has roots at 1, 2, 3
            DoubleUnaryOperator func = x -> (x - 1) * (x - 2) * (x - 3);
            double root = solver.findRoot(func, 0.5, 1.5);
            Assertions.assertEquals(1.0, root, 1E-10);
        }

        @Test
        @DisplayName("should handle function that is zero on entire sub-interval gracefully")
        void findRoot_zeroSubInterval() {
            // Function is exactly zero at x=0
            DoubleUnaryOperator func = x -> x;
            double root = solver.findRoot(func, -1, 1);
            Assertions.assertEquals(0.0, root, 1E-10);
        }

        @ParameterizedTest
        @MethodSource("com.synerset.numenormath.solver.BrentDekkerSolverTest#wideRangeRoots")
        @DisplayName("should find roots across wide range of magnitudes")
        void findRoot_wideRangeOfMagnitudes(double shift, double lowerBound, double upperBound) {
            DoubleUnaryOperator func = x -> x - shift;
            double root = solver.findRoot(func, lowerBound, upperBound);
            Assertions.assertEquals(shift, root, Math.max(1E-10, Math.abs(shift) * 1E-10));
        }
    }

    static Stream<Arguments> wideRangeRoots() {
        return Stream.of(
                Arguments.of(0.0, -1, 1),
                Arguments.of(1E-10, -1, 1),
                Arguments.of(-1E-10, -1, 1),
                Arguments.of(1E4, 0, 2E4),
                Arguments.of(-1E4, -2E4, 0),
                Arguments.of(1E8, 0, 2E8),
                Arguments.of(0.001, -1, 1),
                Arguments.of(-0.001, -1, 1)
        );
    }
}
