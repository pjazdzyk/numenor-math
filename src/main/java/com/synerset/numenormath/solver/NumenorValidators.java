package com.synerset.numenormath.solver;

import com.synerset.numenormath.exception.NumenorSolverException;

class NumenorValidators {

    private NumenorValidators() {
        throw new IllegalStateException("Utility class");
    }

    public static void requireNonInfiniteAndNonNANResults(String name, double... values) {
        for (double num : values) {
            if (Double.isInfinite(num))
                throw new NumenorSolverException(name + ": Solution error. Infinite number detected.");
            if (Double.isNaN(num))
                throw new NumenorSolverException(name + ": Solution error. NaN value detected.");
        }
    }

    public static void requireNonNull(String variableName, Object object) {
        if (object == null) {
            throw new NumenorSolverException("Argument [" + variableName + "] cannot be null.");
        }
    }

    public static boolean containsInfOrNan(double... values) {
        for (double num : values) {
            if (Double.isInfinite(num))
                return true;
            if (Double.isNaN(num))
                return true;
        }

        return false;
    }

}
