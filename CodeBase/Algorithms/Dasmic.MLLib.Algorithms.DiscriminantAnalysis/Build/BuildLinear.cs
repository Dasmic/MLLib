using System.Collections.Generic;
using Dasmic.MLLib.Common.MLCore;


namespace Dasmic.MLLib.Algorithms.DiscriminantAnalysis
{
    
    public class BuildLinear: BuildBase
    {
        public override Common.MLCore.ModelBase BuildModel(double[][] trainingData,
                             string[] attributeHeaders,
                             int indexTargetAttribute)
        {
            //http://people.revoledu.com/kardi/tutorial/LDA/Numerical%20Example.html

            //Verify data and set variables
            VerifyData(trainingData, attributeHeaders, indexTargetAttribute);
            
            ModelLinear model = new ModelLinear(_missingValue,
                                                _indexTargetAttribute,
                                                _trainingData.Length - 1);
            
            //Find number of classes
            HashSet<double> uniqueValues = 
                    GetUniqueValues(trainingData[indexTargetAttribute]);

            model.ClassValues = new double[uniqueValues.Count];
            
            //Split input on classes
            List<double[][]> classInputMatrix =
                            GetClassBasedInputMatrix(uniqueValues,
                                ref model.ClassValues);

            //Get class probabiltities
            model.ClassProbabilities =
                        GetClassProbabilities(classInputMatrix);

            //Compute mean of each attribute in class
            model.ClassMeanMatrix =
                            GetClassMeanMatrix(classInputMatrix);

            //Compute mean of each attribute in total
            double[] dataSetGlobalMeanMatrix =
                            GetDataSetMeanMatrix();

            //Compute difference of classMeanMatrix with entire dataSet mean
            //NOTE: This is different from Jason, in Jason you substract only from mean of the class
            List<double[][]> correctedDataMatrix =
                            GetCorrectedDataMatrix(classInputMatrix,
                                                       dataSetGlobalMeanMatrix);

            //Compute covariance of corrected data matrix
            List<double[][]> covarianceMatrix =
                GetCoVarianceMatrix(correctedDataMatrix);

            //Compute pooled covariance matrix
            double[][] pooledCovarianceMatrix =
                        GetPooledCoVarianceMatrix(covarianceMatrix,
                                                    model.ClassProbabilities);
            //Get inverse of pooled covariance matrix
            model.PooledCovarianceMatrixInv= 
                        _mo.Inverse(pooledCovarianceMatrix);

            return model;
        } //buildModel

       
    
    }
}
