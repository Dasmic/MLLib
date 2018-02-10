using Dasmic.MLLib.Common.Exceptions;
using Dasmic.MLLib.Math.Statistics;
using Dasmic.MLLib.Common.MLCore;


namespace Dasmic.MLLib.Algorithms.Regression
{
    public class BuildLinearSGD:BuildBase
    {
        private double _learningRate;
        private double _initB0;
        private double _initB1;
        private double _maxIterations;

        public BuildLinearSGD():base()
        {

            //Set default values
            _learningRate = .01;
            _maxIterations = 30;  
            _initB0 = 0;
            _initB1 = 0;
        }

        /// <summary>
        /// 0 = Learning Rate
        /// 1 = maxIterations
        /// 2 = initB0
        /// 3 = initB1
        /// </summary>
        /// <param name="values"></param>
        public override void
             SetParameters(params double[] values)
        {
            if(values.Length > 0) _learningRate = values[0];
            if (values.Length > 1) _maxIterations = values[1];
            if (values.Length > 2)  _initB0 = values[2];
            if (values.Length > 3)  _initB1 = values[3];
        }



        //Training Data Should have only Two Attributes
        public override Common.MLCore.ModelBase BuildModel(double[][] trainingData,
                             string[] attributeHeaders,
                             int indexTargetAttribute)
        {
            VerifyData(trainingData, attributeHeaders, indexTargetAttribute);

            double[] x, y;
            ModelLinearBase model = new ModelLinearBase(_missingValue,
                _indexTargetAttribute,_trainingData.Length-1);

            //Additional Checks
            if (trainingData.Length != 2)
                throw new InvalidTrainingDataException();
            if (indexTargetAttribute == 0)
            {
                x = trainingData[1];
                y = trainingData[0];
            }
            else
            {
                x = trainingData[0];
                y = trainingData[1];
            }

            model.B1 = _initB1;// Dispersion.CorrelationPearson(x, y)
                               //* (Dispersion.StandardDeviationSample(y) /
                               //        Dispersion.StandardDeviationSample(x));

            model.B0 = _initB0;// Dispersion.Mean(y) -
                             //_lm.B1 * Dispersion.Mean(x);

            int iter=0;
            int row;
            double predY,error;
            while (iter++ < _maxIterations) //Do for max iterations
            {
                for(row=0;row<x.Length;row++)
                {
                    predY = model.B0 + x[row] * model.B1;
                    error = predY - y[row];
                    model.B0 = model.B0 - error * _learningRate;
                    model.B1 = model.B1 - error * _learningRate * x[row];
                }
            }

            //B0 and B1 values are computed
            return model;
        }
    }
}
