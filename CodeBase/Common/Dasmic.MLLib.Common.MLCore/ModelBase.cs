using Dasmic.MLLib.Common.Exceptions;
using Dasmic.Portable.Core;
using System.Threading.Tasks;
using System;

namespace Dasmic.MLLib.Common.MLCore
{
    public abstract class ModelBase
    {
        protected  double _missingValue;    
        protected int _origTargetAttributeIndex;
        protected int _origAttributeCount;
        protected int _maxParallelThreads;

        protected void VerifyDataForRun(double[] data)
        {
            if (data == null)
                throw new InvalidDataException();

            if (_origAttributeCount !=
                            data.Length)
                throw new AttributeCountMismatchRunModelException();
        }

        public ModelBase(double missingValue,
                            int indexTargetAttribute,
                            int countAttributes)
        {
            SetValues(missingValue, 
                indexTargetAttribute, countAttributes);            
        }

        /// <summary>
        /// For use when value shave to be updated manually
        /// </summary>
        /// <param name="missingValue"></param>
        /// <param name="indexTargetAttribute"></param>
        /// <param name="countAttributes"></param>
        public void SetValues(double missingValue,
                            int indexTargetAttribute,
                            int countAttributes)
        {
            _missingValue = missingValue;
            _origTargetAttributeIndex = indexTargetAttribute;
            _origAttributeCount = countAttributes;
            _maxParallelThreads = -1;//Default value
        }

        //get output
        public abstract double RunModelForSingleData(double[] data);

        //Serialization Routine
        public abstract void SaveModel(string filePath);

        //Deserialization Routine
        public abstract void LoadModel(string filePath);

       
        /// <summary>
        /// Get a single row 2D array for 1D array
        /// 
        /// Useful when a common function is used for runForComputedValues
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        protected double[][] Convert1Dto2D(double[] data)
        {
            double[][] newData = new double[data.Length][];

            for (int col = 0; col < newData.Length; col++)
            {
                newData[col] = new double[1];
                newData[col][0] = data[col];
            }
            return newData;
        }


        /// <summary>
        /// Returns a jagged 2D array with required parameters
        /// 
        /// Array is in the form 2DArray[cols][rows]
        /// </summary>
        /// <param name="cols"></param>
        /// <param name="rows"></param>
        /// <param name="maxThreads"></param>
        /// <returns></returns>
        public double[][]
            Get2DArray(int cols, int rows)
        {
            double[][] array2D  = 
                SupportFunctions.Get2DArray(cols, 
                                    rows, _maxParallelThreads);
            return array2D;
        }

        /// <summary>
        /// Computes the combined Root Mean Square Error of all samples
        /// The trainingData passed should be the same that was used to build the model
        /// 
        /// The error of each sample row of input is squared, then sum is taken
        /// Sum is then divided by the total number of samples and square root taken
        /// </summary>
        /// <param name="trainingData"></param>
        /// <returns></returns>
        public double GetModelRMSE(double[][] trainingData)
        {
            ModelBase model = this;

            //Verify data and set variables
            if (_origTargetAttributeIndex != trainingData.Length - 1)
                throw new TargetAttributeIndexException();

            int inputIdx = 0;
            double rmse = 0;
            Object mutex = new Object();

            if (_origTargetAttributeIndex == 0)
                inputIdx = 1;

            //Do not parallelize as there are some issues
            for(int row=0;row< trainingData[inputIdx].Length;row++)           
            {
                double[] data = GetSingleRowData(trainingData,
                        row, trainingData.Length - 2);
                double output = model.RunModelForSingleData(data);
               
                //Cannot use Interlocked.Add as it is not thread safe
                lock (mutex) //This op is not thread safe
                {
                    rmse += System.Math.Pow(output - trainingData[_origTargetAttributeIndex][row], 2);
                }
            }

            return System.Math.Sqrt(rmse / trainingData[inputIdx].Length);
        }

        /// <summary>
        /// Returns a Single Rows of Data from a 2D array
        /// </summary>
        /// <param name="mainArray"></param>
        /// <param name="mainArrayRowIdx"></param>
        /// <param name="endColumnIdx"></param>
        /// <returns></returns>
        public double[] GetSingleRowData(double[][] mainArray,
                                    long mainArrayRowIdx, int endColumnIdx)
        {         
            double[] data = SupportFunctions.GetLinearArray(mainArray,
                                                mainArrayRowIdx,endColumnIdx);
            return data;
        }
    }
}
