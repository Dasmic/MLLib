using System.Threading.Tasks;
using System.Collections.Generic;
using System.Linq;
using Dasmic.MLLib.Common.Exceptions;
using Dasmic.MLLib.Math.Statistics;
using Dasmic.Portable.Core;
using System;


namespace Dasmic.MLLib.Common.MLCore
{
    public abstract class BuildBase: IBuildModel
    {
        protected double[][] _trainingData;
        protected string[] _attributeHeaders;
        protected int _indexTargetAttribute;
        protected double _missingValue;

        protected int _noOfAttributes;
        protected int _noOfDataSamples;
        protected int _maxParallelThreads;


        public BuildBase()
        {
            _maxParallelThreads = -1; //Set to default
            _missingValue = Constants.MISSING_VALUE;
        }

        #region Support Functions

        protected void VerifyData(double[][] trainingData,
                             string[] attributeHeaders,
                             int indexTargetAttribute)
        {
            _trainingData = trainingData;
            _attributeHeaders = attributeHeaders;
            _indexTargetAttribute = indexTargetAttribute;

            if (_trainingData != null)
            {
                _noOfAttributes = _trainingData.Length - 1;
                _noOfDataSamples = _trainingData[indexTargetAttribute].Length;
            }
            VerifyData();
        }

        /// <summary>
        /// Do not use this inside buildModel
        /// </summary>
        /// <returns></returns>
        protected void VerifyData()
        {
            if (_trainingData.Length !=
                _attributeHeaders.Length)
                throw new AttributesCountMismatchException();

            //Target attribute should be the last column in Training Data
            if (_indexTargetAttribute != _trainingData.Length - 1)
                throw new TargetAttributeIndexException();
        }

        /// <summary>
        /// Generate Unique Random number between a range
        /// </summary>
        /// <param name="min">Start for random number generation</param>
        /// <param name="max">End for random number generation</param>
        /// <param name="count">Number of unique random numbers to be generated</param>
        /// <returns></returns>
        protected int[] GetUniqueRandomNumbers(int min,int max, int count)
        {
            Random rnd = new Random();
            int number;

            HashSet<int> uniqueNumbers = 
                            new HashSet<int>();
            while(uniqueNumbers.Count < count)
            {
                number = rnd.Next(min, max);

                if (!uniqueNumbers.Contains(number)) 
                    uniqueNumbers.Add(number);
            }            
            return uniqueNumbers.ToArray();
        }

        /// <summary>
        /// Returns a new 2D array with specified parameters
        /// </summary>
        /// <param name="columns">Number of columns</param>
        /// <param name="rows">Number of rows</param>
        protected double[][] Get2DArray(long columns, long rows)
        {
            double[][] newArray = new double[columns][];
            //for (int col=0;col<rows;col++)
            Parallel.For(0, columns, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, col =>
            {
                newArray[col] = new double[rows];
            });
            return newArray;
        }
             
        /// <summary>
        /// Gets number of unique values in data array passed to it
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        protected HashSet<double> GetUniqueValues(double[]
                                            data)
        {
            HashSet<double> hashSet = new HashSet<double>();

            foreach (double value in data)
            {
                if (!hashSet.Contains(value))
                    hashSet.Add(value);
            }
            return hashSet;
        }

        /// <summary>
        /// Gets number of unique values in data array passed to it
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        /*protected Dictionary<double,double> getUniqueValuesAndCount(double[]
                                            data)
        {
            HashSet<double> hashSet = new HashSet<double>();

            foreach (double value in data)
            {
                if (!hashSet.Contains(value))
                    hashSet.Add(value);
            }
            return hashSet;
        }*/


        /// <summary>
        /// Returns a list of 2D (attribute X row) matrix 
        /// such that each row in each 2D matrix has the same target
        /// value
        /// </summary>
        /// <param name="uniqueValues"></param>
        /// <param name="classValues"></param>
        /// <returns></returns>
        protected List<double[][]> GetClassBasedInputMatrix(
            HashSet<double> uniqueValues, ref double[] classValues)
        {
            if (classValues == null)
                throw new InvalidDataException();

            List<double[][]> classInputMatrix =
                            new List<double[][]>();

            int classValuesIdx = 0;

            //Do not parallelize this as there is an increment at the end
            foreach (double uv in uniqueValues)
            {
                //Execute Linq query
                var allRows = Enumerable.Range(0, _noOfDataSamples)
                    .Where(row => _trainingData[_indexTargetAttribute][row]
                                    == uv)
                    .ToList();

                double[][] valuesPerRow = new double[_noOfAttributes][];

                //var carMake = carList
                //.Where(item => item.Model == "bmw")
                //.Select(item => item.Make);
                Parallel.For(0, _noOfAttributes, colIdx =>
                {
                    valuesPerRow[colIdx] = new double[allRows.Count];
                });

                //Search all Rows
                Parallel.For(0, allRows.Count, idx =>
                {
                    //Copy values
                    for (int colIdx = 0; colIdx < _noOfAttributes; colIdx++)
                    {
                        //Assume training data is last row
                        valuesPerRow[colIdx][idx] =
                                      _trainingData[colIdx][allRows[idx]];
                    }
                }); //for each row

                classInputMatrix.Add(valuesPerRow); //Should match with classValues
                classValues[classValuesIdx++] = uv; //Should match with classValues
            } //for each unique values

            return classInputMatrix;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="classInputMatrix"></param>
        /// <returns>Returns a list of 2D array.
        /// Even though there is 1 rowm the 2D array is returned
        /// for use in any future matrix computations</returns>
        protected List<double[][]> GetClassMeanMatrix(List<double[][]>
                                            classInputMatrix)
        {
            if (classInputMatrix == null)
                throw new InvalidMatrixException();
            List<double[][]> meanMatrix = new List<double[][]>();

            int noOfAttributes = classInputMatrix[0].Length;
            foreach (double[][] cim in classInputMatrix)
            {
                double[][] mean = new double[noOfAttributes][];

                Parallel.For(0, mean.Length, idx =>
                {
                    mean[idx] = new double[1];
                    mean[idx][0] =
                       Dispersion.Mean(cim[idx]);
                });

                meanMatrix.Add(mean);
            }

            return meanMatrix;
        }


        /// <summary>
        /// Compute the SD of each column
        ///
        /// </summary>
        /// <param name="classInputMatrix"></param>
        /// <returns>Returns a list of 2D array.
        /// Even though there is 1 rowm the 2D array is returned
        /// for use in any future matrix computations</returns>
        protected List<double[][]> GetClassStandardDeviationMatrix(List<double[][]>
                                            classInputMatrix, List<double[][]>
                                            classMeanMatrix)
        {
            if (classInputMatrix == null || classMeanMatrix==null)
                throw new InvalidMatrixException();
            List<double[][]> sdMatrix = new List<double[][]>();

            int noOfAttributes = classInputMatrix[0].Length;
            //foreach (double[][] cim in classInputMatrix)
            for(int ii=0;ii<classInputMatrix.Count;ii++)
            {
                double[][] cim = classInputMatrix[ii];
                double[][] sd = new double[noOfAttributes][];
                
                //For each attribute
                Parallel.For(0, sd.Length, idx =>
                {
                    sd[idx] = new double[1];
                    //Do not call SD functionin Statistic routine
                    //as mean is already known and it will be double computation
                    sd[idx][0] =
                       Dispersion.StandardDeviationPopulation(
                           cim[idx],classMeanMatrix[ii][idx][0]);
                });

                sdMatrix.Add(sd);
            }

            return sdMatrix;
        }

        /// <summary>
        /// Copies a single row of a 2D jagged array into 
        /// a 1D array
        /// </summary>
        /// <param name="mainArray"></param>
        /// <param name="mainArrayRowIdx"></param>
        /// <param name="endColumnIdx"></param>
        /// <returns></returns>
        public double[] GetLinearArray(double [][] mainArray,long mainArrayRowIdx,int endColumnIdx)
        {
            return SupportFunctions.GetLinearArray(mainArray, mainArrayRowIdx, endColumnIdx);           
        }

        /// <summary>
        /// Get sorted values of KeyValuePair in ascending order
        /// </summary>
        /// <param name="allValues"></param>
        /// <returns></returns>
        public List<KeyValuePair<int, double>>
                GetSortedKeyValuePair(KeyValuePair<int, double> [] allValues)
        {
            return SupportFunctions.GetSortedKeyValuePair(
                                allValues);
        }

        #endregion

        
        #region IBuildModel Methods
        public abstract ModelBase BuildModel(double[][] trainingData,
                             string[] attributeHeaders,
                             int indexTargetAttribute);

        /// <summary>
        /// Builds a model where only a single data row is passed at a time
        /// 
        /// This approach is used to conserve memory and can also be used to dynamically
        /// improve the model as new data is available
        /// 
        /// This function assumes that InitializeData has been called first
        /// </summary>
        /// <param name="trainingData"></param>        
        /// <param name="indexTargetAttribute"></param>
        /// <returns></returns>
        public virtual ModelBase BuildModelSingle(double[][][] trainingData,
                        Dictionary<double, string> targetValueMapping) //Mapping between target values and their string values
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Used in conjueciton
        /// </summary>
        /// <param name="targetHeaders">All target values</param>
        public virtual void InitializeModel(string[] targetHeaders)
        {
            throw new NotImplementedException();
        }

        public virtual void
                SetParameters(params double[] values)
        {

        }

        public void SetMissingValue(double value)
        {
            _missingValue = value;
        }
        #endregion

        public int MaxParallelThreads
        {
            get
            {
                return _maxParallelThreads;
            }
            set
            {
                _maxParallelThreads = value;
            }
        }
    }
}
