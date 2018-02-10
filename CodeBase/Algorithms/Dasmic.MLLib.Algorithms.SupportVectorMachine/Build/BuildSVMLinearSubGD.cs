using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.SupportVectorMachine
{
    
    public class BuildSVMLinearSubGD:BuildBase
    {
        private double _learningRate;       
        private double _maxAccuracy; //betwell 0 and 1.0
        private int _checkAccuracyPoint;

        public BuildSVMLinearSubGD():base()
        {
            //Set default values
            _learningRate = .0001;
            _maxIterations = 10000;
            _maxAccuracy = 1;
            _checkAccuracyPoint = 50;
        }

        /// <summary>
        /// 0 = Max accuracy,Range 0 - 1
        /// 1 = Check accuracy point (X). Will check accuracy every Xth iteration
        /// 2 = Learning Rate
        /// 3 = Max Iterations, greater than 0
        /// </summary>
        /// <param name="values"></param>
        public override void
             SetParameters(params double[] values)
        {
            if (values.Length > 0)
                if(values[0] != double.NaN) _maxAccuracy = values[0];
            if (values.Length > 1)
                if (values[1] != double.NaN) _checkAccuracyPoint = (int)values[1];
            if (values.Length > 2)
                if (values[2] != double.NaN) _learningRate = values[2];
            if (values.Length > 3)
                if (values[3] != double.NaN) _maxIterations = (long) values[3];            
        }

        public override Common.MLCore.ModelBase BuildModel(double[][] trainingData,
                             string[] attributeHeaders,
                             int indexTargetAttribute)
        {
            //Verify data and set variables
            VerifyData(trainingData, attributeHeaders, indexTargetAttribute);

            ModelSVMLinearSubGD model = 
                            new ModelSVMLinearSubGD(_missingValue,
                                                _indexTargetAttribute,                   
                                                _trainingData.Length - 1);
            //Set all coeffs to 0
            model.resetCoeffs();        
            int row = 0;
            //Iterate Max Iter
            //Do not parallelize
            //Iter should start from 1 to prevent division by 0 operations
            for(int iter=1;iter<=_maxIterations;iter++) 
            {
                //Iterate per training example            
                //Select a training sample randomly               
                double val=0;
                int col;
                for (col = 0; col < _trainingData.Length - 1; col++)
                {
                    val += model.getCoeff(col) * _trainingData[col][row];
                }
                val += model.getCoeff(col);//Bias term
                val = _trainingData[_indexTargetAttribute][row] * val; //Multiply by Y

                double newVal = 0;
                if (val > 1) //Not a support vector
                {
                    //Update weights             
                    //for (idx = 0; idx < model.getCoeffCount() - 1;
                    //                    idx++)
                    Parallel.For(0, model.getCoeffCount(), new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, idx =>
                    {
                        newVal = model.getCoeff(idx) * (1.0 - (1.0 / iter));
                        model.setCoeff(idx, newVal);
                    });
                    //For bias coeff
                    newVal = model.getCoeff(model.getCoeffCount()-1) * (1.0 - (1.0 / iter));
                    model.setCoeff(model.getCoeffCount() - 1, newVal);
                }
                else //is Support Vector
                {
                    //int idx;
                    //for (idx = 0; idx < model.getCoeffCount() - 1;
                    //                   idx++)
                    Parallel.For(0, model.getCoeffCount(), new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, idx =>
                     {
                         newVal = model.getCoeff(idx) * (1.0 - (1.0 / iter));
                         newVal += newVal + 1.0 / (_learningRate * iter) * _trainingData[idx][row] *
                                                                     _trainingData[_indexTargetAttribute][row];
                         model.setCoeff(idx, newVal);
                     });                   
                    //For bias coeff
                    newVal = model.getCoeff(model.getCoeffCount() - 1) * (1.0 - (1.0 / iter));
                    newVal += 1.0 / (_learningRate * iter) * 1.0 *
                                               _trainingData[_indexTargetAttribute][row];
                    model.setCoeff(model.getCoeffCount() - 1, newVal);
                }
                //Check accuracy every 50 iterations
                if (iter % _checkAccuracyPoint == 0)
                    if (model.getAccuracy(_trainingData) >= _maxAccuracy)
                        break;//We have reached our threshold. No need to go further
                //Goto next row
                row++;
                row = row > (_noOfDataSamples - 1) ? 0 : row;
            }//For each epoch            
            return model;

        }        
    }
}
