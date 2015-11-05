
import MultivariateLinearRegression.MultivariateLR;
import MultivariateLinearRegression.GradientDescentValues;
import Jama.Matrix;

class LogReg extends MultivariateLR{
	


	/**
     * GRADIENTDESCENTMULTI Performs gradient descent to learn theta
        theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
        taking num_iters gradient steps with learning rate alpha
     * @param X
     * @param y
     * @param theta
     * @param alpha
     * @param numIterations
     * @return 
     */
    @Override
    public GradientDescentValues gradientDescent(Matrix X, Matrix y, Matrix theta, double alpha, int numIterations){
        //Write equivalent Java code for the Octave code below.
        
        //Initialize some useful values.
        //Octave: m = length(y); % number of training examples

        //create a matrix that stores cost history
        //Octave: J_history = zeros(num_iters, 1);

        //Loop thru numIterations
        //Octave:for iter = 1:num_iters
        
        // ====================== YOUR CODE HERE ======================
        // Instructions: Perform a single gradient step on the parameter vector
        //               theta. 
        //
        // Hint: While debugging, it can be useful to print out the values
        //       of the cost function (computeCostMulti) and gradient here.
        //

        
        
        
        
        
        
        
        
        
        // Save the cost J in every iteration    
        //Octave: J_history(iter) = costFunction(theta, X, y);
        return null;
    }

    /**
		%COSTFUNCTION Compute cost and gradient for logistic regression
		%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
		%   parameter for logistic regression and the gradient of the cost
		%   w.r.t. to the parameters.
	*/
    protected double costFunction(Matrix theta, Matrix X, Matrix y){
            /**
                    % Initialize some useful values
                    m = length(y); % number of training examples

                    % You need to return the following variables correctly 
                    J = 0;
                    grad = zeros(size(theta));

                    % ====================== YOUR CODE HERE ======================
                    % Instructions: Compute the cost of a particular choice of theta.
                    %               You should set J to the cost.
                    %               Compute the partial derivatives and set grad to the partial
                    %               derivatives of the cost w.r.t. each parameter in theta
                    %
                    % Note: grad should have the same dimensions as theta
                    %
                    % h = compute hypothesis
                    % J = cost function
                    %
            */


    }


    /**
            %COSTFUNCTION Compute cost and gradient for logistic regression
            %   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
            %   parameter for logistic regression and the gradient of the cost
            %   w.r.t. to the parameters.
    */
  

    /*
    function g = sigmoid(z)
    %SIGMOID Compute sigmoid functoon
    %   J = SIGMOID(z) computes the sigmoid of z.
    */
    double sigmoid(Matrix z){
            /*
            % You need to return the following variables correctly 
            g = zeros(size(z));

            % ====================== YOUR CODE HERE ======================
            % Instructions: Compute the sigmoid of each value of z (z can be a matrix,
            %               vector or scalar).

            g = 1./(1+exp(-z));
            % ============================================================
            */




    }

    /*
    function p = predict(theta, X)
    %PREDICT Predict whether the label is 0 or 1 using learned logistic 
    %regression parameters theta
    %   p = PREDICT(theta, X) computes the predictions for X using a 
    %   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    */
    Matrix predict(Matrix theta, Matrix X){
            /*
            m = size(X, 1); % Number of training examples

            % You need to return the following variables correctly
            p = zeros(m, 1);

            % ====================== YOUR CODE HERE ======================
            % Instructions: Complete the following code to make predictions using
            %               your learned logistic regression parameters. 
            %               You should set p to a vector of 0's and 1's
            %
            for i=1:m
              h = sigmoid(X(i,:)*theta)
              if h >= 0.5
                 p(i) = 1;
              else
                 p(i) = 0;
              end
            end
            % =========================================================================
            */






    }


    /**
            fprintf('Train Accuracy: %f\n', );		
    */
    double accuracy(Matrix p, Matrix y){
            //mean(double(p == y)) * 100



    }



    public static void main(String[] args){
            /**
                    %% Load Data
                    %  The first two columns contains the exam scores and the third column
                    %  contains the label.

                    data = load('ex2data1.txt');
                    X = data(:, [1, 2]); y = data(:, 3);
            */
            //===== JAVA CODE HERE ====



            /*
            [X mu sigma] = featureNormalize(X);
            */
            //===== JAVA CODE HERE ====




            /**
                    plotData(X, y);

                    % Put some labels 
                    hold on;
                    % Labels and Legend
                    xlabel('Exam 1 score')
                    ylabel('Exam 2 score')

                    % Specified in plot order
                    legend('Admitted', 'Not admitted')
            */
            //===== JAVA CODE HERE ====




            /**

                    Compute Cost and Gradient
                    %  Setup the data matrix appropriately, and add ones for the intercept term
                    [m, n] = size(X);

                    % Add intercept term to x and X_test
                    X = [ones(m, 1) X];

                    % Initialize fitting parameters
                    initial_theta = zeros(n + 1, 1);

                    % Compute and display initial cost and gradient
                    [cost, grad] = costFunction(initial_theta, X, y);

                    fprintf('Cost at initial theta (zeros): %f\n', cost);
                    fprintf('Gradient at initial theta (zeros): \n');
                    fprintf(' %f \n', grad);

                    Cost at initial theta (zeros): 0.693147
                    Gradient at initial theta (zeros):
                     -0.100000
                     -12.009217
                     -11.262842

            */
            //===== JAVA CODE HERE ====








            /**
                    Optimizing using GD
            % Choose some alpha value
            alpha = 0.1;
            num_iters = 1000;

            % Init Theta and Run Gradient Descent 
            theta = zeros(3, 1);
            [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

            % Plot the convergence graph
            figure;
            plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
            xlabel('Number of iterations');
            ylabel('Cost J');

            % Display gradient descent's result
            fprintf('Theta computed from gradient descent: \n');
            fprintf(' %f \n', theta);
            */ 
            */

        //===== JAVA CODE HERE ====





        /**
        Predict and Accuracies
        prob = sigmoid([1 45 85] * theta);
            fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
                     'probability of %f\n\n'], prob);

            % Compute accuracy on our training set
            p = predict(theta, X);

        */
            //===== JAVA CODE HERE ====


            /**
            fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
            */
            //===== JAVA CODE HERE ====




    }	
}
