 using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Dasmic.MLLib.Math.Matrix;

namespace Dasmic.MLLib.Algorithms.Regression
{ 
    /// <summary>
    /// Multivariable Regression Model
    /// 
    /// This is the case when there is one response variable
    /// and multiple predictor variables
    /// 
    /// Note that trainingData values will be modified by this algorithm so 
    /// do not use the Training Data later in yoru program. This saves
    /// a depp copy of the training data matrix
    /// 
    /// General formula:
    /// Beta = cov(X,Y) / var(X)
    /// 
    /// Beta = (X'X)^-1 * X' * Y
    /// 
    /// Beta is a matrix of coefficients
    /// X'X/n = covariance matrix of X
    /// 
    /// Ref: http://reliawiki.org/index.php/Multiple_Linear_Regression_Analysis
    /// </summary>
    public class BuildLinearMultiVariable : BuildBase
    {
        public override Common.MLCore.ModelBase BuildModel(double[][] trainingData,
                            string[] attributeHeaders,
                            int indexTargetAttribute)
        {
            VerifyData(trainingData, attributeHeaders, indexTargetAttribute);
            
            ModelLinearMultiVariableBase model = new ModelLinearMultiVariableBase(_missingValue,
                                        _indexTargetAttribute, _trainingData.Length - 1);

            
            //Create Matrix Y and replace Target values with 1
            double[][] Y = new double[1][];
            Y[0] = new double[trainingData[0].Length];
            Parallel.For(0, Y[0].Length,
                   new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },
                   row =>
            {
                Y[0][row] = trainingData[indexTargetAttribute][row];
                trainingData[indexTargetAttribute][row] = 1;//Set it to 
            });

            MatrixOperations mo = new MatrixOperations();

            double[][] transposeX = mo.Transpose(trainingData);
            double[][] multiply = mo.Multiply(transposeX, trainingData);
            double[][] inverse = mo.Inverse(multiply);

            double[][] final = mo.Multiply(inverse, transposeX);
            final = mo.Multiply(final, Y);

            //Copy coeffs into main array
            //Intercept is in last index
            for (int idx = 0; idx < _noOfAttributes + 1; idx++)
                model.SetCoeff(idx, final[0][idx]);

            //Restore Y values in TrainingData
            Parallel.For(0, Y[0].Length,
                new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },
                row =>
                {
                    trainingData[indexTargetAttribute][row] = Y[0][row];//Set it to 
                });

            return model;
        }
    }
}
