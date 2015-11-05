package util;


import Jama.Matrix;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author rjesteban
 */
public final class Util {    
    
    
    public static void fprintfMatrix(Matrix X, int rows){
        for(int r=0;r<rows;r++){
            for(int c=0;c<X.getColumnDimension();c++){
                System.out.print(X.get(r, c) + " ");
            }
            System.out.println();
        }
    }
    
    public static void fprintfMatrix(Matrix X, Matrix Y, int rows){
        for(int r=0;r<rows;r++){
            for(int c=0;c<X.getColumnDimension();c++){
                System.out.print(X.get(r, c) + " ");
            }
            System.out.println(" " + Y.get(r, 0));
        }
    }
    
    public static Matrix insertX0(Matrix F){
        Matrix _X = new Matrix(F.getRowDimension(),F.getColumnDimension()+1);
        for(int r=0;r<F.getRowDimension();r++){
            for(int c=0;c<=F.getColumnDimension();c++){
                if(c==0)
                    _X.set(r, c, 1);
                else
                    _X.set(r, c, F.get(r, c-1));
            }
        }
        return _X;
    }

    /**
     * data returns vector data set with specified column
     * file: data file
     * col: starting index
     * @param filename
     * @param start
     * @return Matrix
     */
    public static Matrix data(String filename, int col) throws FileNotFoundException, IOException {
        BufferedReader bf = new BufferedReader(new FileReader(filename));
        col--;
        String line = bf.readLine();
        String colval = "";
        ArrayList<Double> training_examples = new ArrayList<Double>();

        while (line != null) {
            colval = line.split(",")[col];
            training_examples.add(Double.valueOf(colval));
            line = bf.readLine();
        }
        
        int r = training_examples.size();
        Matrix matrix = new Matrix(r, 1);
        
        for (int i = 0; i < training_examples.size(); i++) 
            matrix.set(i, 0, training_examples.get(i));

        return matrix;
    }

    /**
     * data returns data set using matrix with specified column
     * file: data file
     * start: starting index
     * end: end index
     * @param filename
     * @param start
     * @return Matrix
     */
    public static Matrix data(String filename, int start, int end) throws FileNotFoundException, IOException {
        start--; //index's sake
        end--;
        
        BufferedReader bf = new BufferedReader(new FileReader(filename));
        String line = bf.readLine();
        String[] field = line.split(",");
        ArrayList<double[]> training_examples = new ArrayList<double[]>();
        
        while (line != null) {
            field = line.split(",");
            double[] data = new double[(end-start)+1];
            
            for (int i = start; i <= end-start; i++) {
                data[i] = Double.parseDouble(field[i]);
            }
            
            training_examples.add(data);
            line = bf.readLine();
        }

        int c = training_examples.get(0).length;
        int r = training_examples.size();

        Matrix matrix = new Matrix(r, c);
        for (int rr = 0; rr < r; rr++) {
            for (int cc = 0; cc < c; cc++) {
                matrix.set(rr, cc, training_examples.get(rr)[cc]);
            }
        }

        return matrix;
    }
    
}