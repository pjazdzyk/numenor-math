package com.synerset.numenormath.solver;

import com.synerset.numenormath.exception.NumenorSolverException;

import java.util.function.DoubleUnaryOperator;

/**
 * Common interface for numerical root-finding solvers.
 * Provides the minimal shared API for finding roots of single-variable equations.
 * <br>
 * <p>AUTHOR: Piotr Jażdżyk, MScEng</p>
 * CONTACT:
 * <a href="https://www.linkedin.com/in/pjazdzyk">LinkedIn</a> |
 * <a href="http://synerset.com/">www.synerset.com</a><br>
 */
public interface MathSolver {

    /**
     * Finds a root of the function that was previously set on the solver instance.
     *
     * @return the calculated root value
     * @throws NumenorSolverException if the solver cannot converge to a solution
     */
    double findRoot();

    /**
     * Finds a root of the provided function.
     *
     * @param func the function to find a root for (use lambda expression or method reference)
     * @return the calculated root value
     * @throws NumenorSolverException if the solver cannot converge to a solution
     */
    double findRoot(DoubleUnaryOperator func);

    /**
     * Sets the function to be solved.
     *
     * @param func the function to compute (use lambda expression or method reference)
     */
    void setFunction(DoubleUnaryOperator func);

    /**
     * Returns the current accuracy setting.
     *
     * @return accuracy level
     */
    double getAccuracy();

    /**
     * Sets the accuracy for the root-finding algorithm.
     *
     * @param accuracy the desired accuracy level
     */
    void setAccuracy(double accuracy);

    /**
     * Returns the current iteration limit.
     *
     * @return maximum number of iterations
     */
    double getIterationsLimit();

    /**
     * Sets the maximum number of iterations.
     *
     * @param iterationsLimit the iteration limit
     */
    void setIterationsLimit(double iterationsLimit);

    /**
     * Returns the number of iterations performed in the last solve.
     *
     * @return iteration count
     */
    int lastResultIterationCount();

    /**
     * Resets the solver run flags and iteration counter for reuse.
     */
    void resetSolver();

}
