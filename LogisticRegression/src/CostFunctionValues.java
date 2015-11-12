
import Jama.Matrix;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author rjesteban
 */
public class CostFunctionValues {
    private Matrix J;
    private Matrix grad;
    // add getters and setters     

    /**
     * @return the J
     */
    public Matrix getJ() {
        return J;
    }

    /**
     * @param J the J to set
     */
    public void setJ(Matrix J) {
        this.J = J;
    }

    /**
     * @return the grad
     */
    public Matrix getGrad() {
        return grad;
    }

    /**
     * @param grad the grad to set
     */
    public void setGrad(Matrix grad) {
        this.grad = grad;
    }
}
