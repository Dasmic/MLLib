using System;
using System.Collections.Generic;
using Dasmic.MLLib.Common.Exceptions;
using Dasmic.MLLib.Common.MLCore;


namespace Dasmic.MLLib.Algorithms.Regression
{
    /// <summary>
    /// Logistic Regression based on Stoicastic Gradient Descent
    /// </summary>
    public class BuildLogisticSGD:BuildBase
    {
        private double _learningRate;
        private double _maxIterations;

        
        public BuildLogisticSGD():base()
        {

            //Set default values
            _learningRate = .3;
            _maxIterations = 10;
            
        }

        /// <summary>
        /// 0 = Learning Rate
        /// 1 = maxIterations      
        /// </summary>
        /// <param name="values"></param>
        public override void
             SetParameters(params double[] values)
        {
            if (values.Length > 0) _learningRate = values[0];
            if (values.Length > 1) _maxIterations = values[1];
        }

        private void initCoefficients(ModelLogisticBase model,
                                        int noOfAttributes)
        {
            model.B0 = 0;
            model.B = new List<double>();
            //Add Coeff for each attributes
            for(int idx=0;idx<noOfAttributes;idx++)
            {
                double b = 0.0;
                model.B.Add(b);
            }
        }


        /// <summary>
        /// Training Data can have any number of attributes 
        /// </summary>
        /// <param name="trainingData"></param>
        /// <param name="attributeHeaders"></param>
        /// <param name="indexTargetAttribute"></param>
        public override Common.MLCore.ModelBase BuildModel(double[][] trainingData,
                             string[] attributeHeaders,
                             int indexTargetAttribute)
        {
            VerifyData(trainingData, 
                attributeHeaders, 
                indexTargetAttribute);

            ModelLogisticBase  model = 
                new ModelLogisticBase(_missingValue, _indexTargetAttribute, 
                                        _trainingData.Length - 1);
            //Additional Checks
            if (trainingData.Length < 2)
                throw new InvalidTrainingDataException();

            //Also create a coeff for Target Attributes,m just dont use it.
            initCoefficients(model,trainingData.Length);

            
            int iter = 0;
            int row,col;
            double predY=0, pow=0;
            while (iter++ < _maxIterations) //Do for max iterations
            {
                for (row = 0; row < trainingData[0].Length; row++) //For each row of training data
                {
                    pow = 0;
                    //For each training attributes
                    for (col = 0; col < trainingData.Length; col++)
                    {
                        if (col != _indexTargetAttribute)
                        {
                            pow += model.B[col] * trainingData[col][row];
                        }
                    }
                    pow += model.B0;
                    predY = 1 / (1 + System.Math.Exp(-pow));

                    //Now update coeffs
                    model.B0 = model.B0 + getNewValue(1.0,
                                        trainingData[_indexTargetAttribute][row],
                                        predY);
                    for (col = 0; col < trainingData.Length; col++)
                    {
                        //b = b + alpha  (y 􀀀 prediction)  prediction  (1 􀀀 prediction)  x
                        if (col != _indexTargetAttribute)
                        {
                            model.B[col] = model.B[col] + 
                                        getNewValue(trainingData[col][row], 
                                        trainingData[_indexTargetAttribute][row], 
                                        predY);
                        }
                    }                      
                } //For each row
            } //For max iterations

            return model;      
        } //buildModel


        private double getNewValue(double trainingData,double Y, double predY)
        {
            return ((_learningRate *
            (Y - predY)) * predY * (1 - predY) * trainingData);
        }
    }
}
