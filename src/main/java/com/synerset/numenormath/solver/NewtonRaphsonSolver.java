package com.synerset.numenormath.solver;

import java.util.function.DoubleUnaryOperator;
import java.util.logging.Level;
import java.util.logging.Logger;

import static com.synerset.numenormath.solver.NumenorValidators.requireNonInfiniteAndNonNANResults;
import static com.synerset.numenormath.solver.NumenorValidators.requireNonNull;

/**
 * NEWTON-RAPHSON ITERATIVE SOLVER<br>
 * Single variable equation solver for finding roots using the Newton-Raphson method.
 * The algorithm uses the iterative formula: x_{n+1} = x_n - f(x_n) / f'(x_n)
 * where f'(x_n) is approximated numerically using central difference method.
 * <br>
 * The Newton-Raphson method converges quadratically near the root, but requires
 * a good initial guess and may fail if the derivative is zero or near-zero.
 * This implementation includes safeguards against division by zero and divergence.
 * <br>
 * User function must be rearranged to the form of [expression = 0].
 * <br>
 * <p>AUTHOR: Piotr Jażdżyk, MScEng</p>
 * CONTACT:
 * <a href="https://www.linkedin.com/in/pjazdzyk">LinkedIn</a> |
 * <a href="http://synerset.com/">www.synerset.com</a><br>
 */
public class NewtonRaphsonSolver implements MathSolver {

    // DEFAULTS:
    public static final String DEF_NAME = "DefaultBrentSolver"; // default name
    public static final double DEF_A0 = -50;                    // initial guess or arbitrarily assumed value to get opposite (negative and positive) result from tested equation
    public static final double DEF_B0 = 50;                     // initial guess or arbitrarily assumed value to get opposite (negative and positive) result from tested equation
    public static final double DEF_ITERATIONS = 100;            // default limit of iterations
    public static final double DEF_ACCURACY = 1E-11;            // expected accuracy level

    // PROPERTIES

    private final String name;

    private double x0;                         // initial guess
    private double x, f_x;                     // current point and its function value f(x)
    private double xPrev, f_xPrev;             // previous iteration point and f(x)
    private double derivative;                 // numerical derivative f'(x)
    private int counter;

    private DoubleUnaryOperator userFunction;  // function to compute, to be provided by user

    private boolean runFlag;
    private double iterationsLimit;
    private double accuracy;

    private double derivativeStep;             // step size for numerical derivative approximation

    private boolean showDiagnostics;
    private boolean showSummary;

    private boolean failForNaN;

    private static final Logger LOGGER = Logger.getLogger(NewtonRaphsonSolver.class.getName());

    /**
     * Initializes solver instance with function output set as 0, with default name.
     */
    public NewtonRaphsonSolver() {
        this(DEF_NAME, x -> 0, DEF_A0);
    }

    /**
     * Initializes solver instance with function output set as 0, with specified name
     */
    public NewtonRaphsonSolver(String name) {
        this(name, x -> 0, DEF_A0);
    }

    /**
     * Initializes solver instance with a name, user provided function and initial guess.
     *
     * @param name              solver name
     * @param functionToCompute user function representing an equation to be computed
     * @param x0                initial guess for the root
     */
    public NewtonRaphsonSolver(String name, DoubleUnaryOperator functionToCompute, double x0) {
        requireNonNull("functionToCompute", functionToCompute);
        this.name = name == null ? DEF_NAME : name;
        this.userFunction = functionToCompute;
        this.x0 = x0;
        this.runFlag = true;
        this.iterationsLimit = DEF_ITERATIONS;
        this.accuracy = DEF_ACCURACY;
        this.derivativeStep = 1E-8;
        this.failForNaN = true;
    }

    // Calculation methods

    /**
     * Root finding algorithm using Newton-Raphson method. Returns root value.
     * Uses numerical derivative approximation (central difference).
     *
     * @return actual root value meeting accuracy criteria
     */
    public double findRoot() {
        long startTime = System.nanoTime();

        // Initializing Solver input variables
        initializeAndCheckConditions();

        /*--------BEGINNING OF ITERATIVE LOOP--------*/
        log("Starting calculations....");
        logCurrentSolutionStatus("INITIAL VALUES: ", "");
        while (runFlag) {
            counter++;

            // Store previous values
            xPrev = x;
            f_xPrev = f_x;

            // Calculate numerical derivative using central difference
            derivative = computeDerivative(x);

            // Check for zero or near-zero derivative
            if (Math.abs(derivative) < 1E-15) {
                log("WARNING: Near-zero derivative detected ({0}). Switching to secant-like step.", derivative);
                // Fallback: use a small step in the direction of the function
                if (f_x > 0) {
                    x = x - accuracy * 10;
                } else {
                    x = x + accuracy * 10;
                }
            } else {
                // Newton-Raphson iteration: x_{n+1} = x_n - f(x_n) / f'(x_n)
                x = x - f_x / derivative;
            }

            // Evaluate function at new point
            f_x = userFunction.applyAsDouble(x);

            logCurrentSolutionStatus(String.format("ITERATION: %-5s", counter),
                    String.format("f'(x) = %.6f", derivative));

            // Checking conditions if they meet solution criteria
            if (Math.abs(f_x) < accuracy) {
                log("Solver stopped, calculated solution meets accuracy requirement of {0}.", String.valueOf(accuracy));
                runFlag = false;
            } else if (Math.abs(x - xPrev) < accuracy) {
                log("Solver stopped, change in x is below accuracy threshold.");
                runFlag = false;
            } else if (counter > iterationsLimit) {
                runFlag = false;
                log("Solver stopped, reached iterations limit of {0}. Solution is not converged.", iterationsLimit);
            } else if (f_x == 0) {
                runFlag = false;
            }

            // Check for divergence
            if (counter > 5 && Math.abs(f_x) > Math.abs(f_xPrev) * 10) {
                log("WARNING: Possible divergence detected. Function value increasing rapidly.");
            }

            // Exception will be thrown if NaN or Infinite values are detected
            if (failForNaN) {
                requireNonInfiniteAndNonNANResults(name, f_x, x, derivative);
            } else if (NumenorValidators.containsInfOrNan(f_x, x, derivative)) {
                runFlag = false;
                log("Solver stopped, NaN or Inf result detected. Solution is not converged.");
            }

            /*-----------END OF ITERATIVE LOOP-----------*/
        }

        long endTime = System.nanoTime();
        long durationNano = endTime - startTime;
        double durationMillis = durationNano / 1_000_000.0;
        double durationSeconds = durationNano / 1_000_000_000.0;

        logSummary("CALCULATIONS COMPLETE: Root found: [{0}] in {1} iterations. Completed in: {2} millis or {3} seconds. Target accuracy: {4}.",
                String.valueOf(x), counter, durationMillis, durationSeconds, String.valueOf(accuracy));

        return x;
    }

    /**
     * Returns a function root (calculation result) based on provided user-function.
     *
     * @param func tested function (use lambda expression or method reference)
     * @return function root (Double)
     */
    @Override
    public double findRoot(DoubleUnaryOperator func) {
        setFunction(func);
        return findRoot();
    }

    /**
     * Returns a function root (calculation result) based on provided user-function and initial guess.
     *
     * @param func tested function (use lambda expression or method reference)
     * @param x0   initial guess for the root
     * @return function root (Double)
     */
    public double findRoot(DoubleUnaryOperator func, double x0) {
        setCounterpartPoints(x0, x0);  // For Newton-Raphson, we only need one point
        return findRoot(func);
    }

    /**
     * Returns a function root (calculation result) based on provided user-function and two points.
     * For Newton-Raphson, the two points are averaged to produce the initial guess.
     *
     * @param func tested function (use lambda expression or method reference)
     * @param a0   first point (averaged with b0 for initial guess)
     * @param b0   second point (averaged with a0 for initial guess)
     * @return function root (Double)
     */
    public double findRoot(DoubleUnaryOperator func, double a0, double b0) {
        setCounterpartPoints(a0, b0);
        return findRoot(func);
    }

    // Solver control

    /**
     * Resets flags and iteration counter
     */
    @Override
    public void resetSolver() {
        this.runFlag = true;
        this.counter = 0;
    }

    public boolean isFailForNaN() {
        return failForNaN;
    }

    public void setFailForNaN(boolean failForNaN) {
        this.failForNaN = failForNaN;
    }

    // Getters, setters

    /**
     * Sets function to be solved.
     *
     * @param func tested function (use lambda expression or method reference)
     */
    @Override
    public void setFunction(DoubleUnaryOperator func) {
        this.userFunction = func;
        resetSolver();
    }

    /**
     * Sets initial guess for the root using two points (averaged).
     * This method is specific to Newton-Raphson and not part of the MathSolver interface.
     *
     * @param pointA - first point (averaged with pointB for initial guess)
     * @param pointB - second point (averaged with pointA for initial guess)
     */
    public void setCounterpartPoints(double pointA, double pointB) {
        // For Newton-Raphson, we use the average as the initial guess
        this.x0 = (pointA + pointB) / 2.0;
        resetSolver();
    }

    /**
     * Sets the initial guess for the root.
     *
     * @param x0 initial guess
     */
    public void setInitialGuess(double x0) {
        this.x0 = x0;
        resetSolver();
    }

    /**
     * Returns Newton-Raphson accuracy.
     *
     * @return accuracy level
     */
    @Override
    public double getAccuracy() {
        return accuracy;
    }

    /**
     * Sets accuracy if other than default is required.
     *
     * @param accuracy accuracy
     */
    @Override
    public void setAccuracy(double accuracy) {
        this.accuracy = Math.abs(accuracy);
    }

    /**
     * Returns current iteration limit value.
     *
     * @return max iteration limit (int)
     */
    @Override
    public double getIterationsLimit() {
        return iterationsLimit;
    }

    /**
     * Sets maximum iteration limit
     *
     * @param iterationsLimit maximum iteration limit
     */
    @Override
    public void setIterationsLimit(double iterationsLimit) {
        this.iterationsLimit = iterationsLimit;
    }

    /**
     * Returns number of iterations.
     *
     * @return iteration count
     */
    @Override
    public int lastResultIterationCount() {
        return counter;
    }

    /**
     * Returns the step size used for numerical derivative approximation.
     *
     * @return derivative step size
     */
    public double getDerivativeStep() {
        return derivativeStep;
    }

    /**
     * Sets the step size for numerical derivative approximation.
     * Smaller values give more accurate derivatives but may suffer from round-off errors.
     *
     * @param derivativeStep step size for derivative calculation (default: 1E-8)
     */
    public void setDerivativeStep(double derivativeStep) {
        this.derivativeStep = derivativeStep;
    }

    // Initial conditions initializer

    private void initializeAndCheckConditions() {
        log("Starting condition evaluation procedure.");

        x = x0;
        f_x = userFunction.applyAsDouble(x);

        // In case provided by user point is actually a root
        if (accuracyRequirementIsMet()) {
            log("CALCULATION COMPLETE. Initial value is a root: {}.", x);
            runFlag = false;
            return;
        }

        log("Condition evaluation successful.");
    }

    // Numerical derivative computation

    /**
     * Computes the numerical derivative using central difference method.
     * f'(x) ≈ (f(x + h) - f(x - h)) / (2h)
     *
     * @param x point at which to compute the derivative
     * @return approximate derivative value
     */
    private double computeDerivative(double x) {
        double h = derivativeStep;
        double fPlus = userFunction.applyAsDouble(x + h);
        double fMinus = userFunction.applyAsDouble(x - h);
        return (fPlus - fMinus) / (2 * h);
    }

    // Solution state requirements

    private boolean accuracyRequirementIsMet() {
        return Math.abs(f_x) <= accuracy;
    }

    // Debugging solution with diagnostic output

    /**
     * Sets diagnostic output mode. (true = show output, false = no output).
     *
     * @param showDebugLogs true = on, false = off
     */
    public void toggleDebugLogs(boolean showDebugLogs) {
        this.showDiagnostics = showDebugLogs;
    }

    /**
     * Sets solver calculations summary. (true = show output, false = no output).
     *
     * @param showSummaryLogs true = on, false = off
     */
    public void toggleSummaryLogs(boolean showSummaryLogs) {
        this.showSummary = showSummaryLogs;
    }

    // Loggers and diagnostic outputs

    private void log(String msg, Object... msgParams) {
        log(Level.INFO, msg, msgParams);
    }

    private void log(Level level, String msg, Object... msgParams) {
        if (showDiagnostics) {
            LOGGER.log(level, String.format("[%s] - %s", name, msg), msgParams);
        }
    }

    private void logSummary(String msg, Object... msgParams) {
        if (showSummary || showDiagnostics) {
            LOGGER.log(Level.INFO, String.format("[%s] - %s", name, msg), msgParams);
        }
    }

    private void logCurrentSolutionStatus(String titleMsg, String endMsg) {
        String formattedMsg = String.format("%s x= %.5f, f(x)= %.5f, x_prev= %.5f, f(x_prev)= %.5f %s",
                titleMsg, x, f_x, xPrev, f_xPrev, endMsg);

        log(formattedMsg);
    }

    // Static factory methods

    /**
     * Method for obtaining quick and single result for a provided function and initial guess.
     *
     * @param func provided eqn = 0 as an lambda expression: value -> f(value)
     * @param x0   initial guess for the root
     * @return calculated root
     */
    public static double findRootOf(DoubleUnaryOperator func, double x0) {
        NewtonRaphsonSolver solver = NewtonRaphsonSolver.of(func, x0);
        return solver.findRoot();
    }

    public static NewtonRaphsonSolver of() {
        return new NewtonRaphsonSolver();
    }

    public static NewtonRaphsonSolver of(String name) {
        return new NewtonRaphsonSolver(name, x -> 0, DEF_A0);
    }

    public static NewtonRaphsonSolver of(DoubleUnaryOperator functionToCompute, double x0) {
        return new NewtonRaphsonSolver(DEF_NAME, functionToCompute, x0);
    }
}
