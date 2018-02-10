  using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.NearestNeighbour
{
    public class BuildLVQ : BuildBase
    {
        private double _alpha;
        private double _learningRate;
        private long _noOfEpoch;
        private long _noOfNeurons;

        public BuildLVQ()
        {
            _alpha = .3;
            _noOfEpoch = 100;
            _noOfNeurons = long.MaxValue;
            setDistanceMeasure(new DistanceMeasureEuclidean());
        }


        public void setDistanceMeasure(IDistanceMeasure distanceMeasure)
        {
            if (distanceMeasure != null)
                _distanceMeasure = distanceMeasure;
        }

        /// <summary>
        /// Sets parameters
        /// Alpha, Number of Epoch, Number of Neurons
        /// </summary>
        /// <param name="values"></param>
        public override void
              SetParameters(params double[] values)
        {
            if (values.Length > 0) _alpha = values[0];
            if (values.Length > 1) _noOfEpoch = (long)values[1];
            if (values.Length > 2) _noOfNeurons = (long)values[1];
        }

        public override Common.MLCore.ModelBase
            BuildModel(double[][] trainingData,
                     string[] attributeHeaders,
                     int indexTargetAttribute)
        {
            //Verify data and set variables
            VerifyData(trainingData, attributeHeaders, indexTargetAttribute);

            //Get unique value in Target Attribute
            HashSet<double> uniqueTargetValues =
                    GetUniqueValues(trainingData[indexTargetAttribute]);

            if (_noOfNeurons == long.MaxValue)
                _noOfNeurons = uniqueTargetValues.Count * 3; //Max 3 Neurons for each classfication type

            //cbVector = Neurons
            //each attribute is weight
            double[][] cbVector = new double[_trainingData.Length][];
            for (int ii = 0; ii < cbVector.Length; ii++)
                cbVector[ii] = new double[_noOfNeurons];

            //Initialize the code book vector ---------------
            //Get class based matrix then take center, min and max value
            double [] targetValues = new double[uniqueTargetValues.Count];
            List<double[][]> classInputMatrix =
                            GetClassBasedInputMatrix(uniqueTargetValues,
                                ref targetValues);
            int tdRow = 0;
            //for (double[][] classIM in classInputMatrix)
            for(int cimIdx=0;cimIdx<classInputMatrix.Count; cimIdx++)
            {
                double[][] classIM = classInputMatrix[cimIdx];
                if (classIM != null)
                {
                    int row1 = -1, row2 = -1, row3 = -1;
                    row1 = 0;
                    if (classIM[0].Length > 1) row2 = classIM[0].Length-1;
                    if (classIM[0].Length > 2) row3 = classIM[0].Length / 2;

                    //Copy values
                    
                    //for(col = 0;col < classIM.Length;col++)
                    Parallel.For(0, classIM.Length, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, cCol =>
                    {
                        cbVector[cCol][tdRow] = classIM[cCol][row1];
                        if (row2 > 0) cbVector[cCol][tdRow + 1] = classIM[cCol][row2];
                        if (row3 > 0) cbVector[cCol][tdRow + 2] = classIM[cCol][row3];
                    });

                    //Add last column for values
                    int col = cbVector.Length-1;
                    cbVector[col][tdRow] = targetValues[cimIdx];
                    if (row2 > 0) cbVector[col][tdRow + 1] = targetValues[cimIdx];
                    if (row3 > 0) cbVector[col][tdRow + 2] = targetValues[cimIdx];

                    //Increment row
                    tdRow++;
                    if (row2 > 0) tdRow++;
                    if (row3 > 0) tdRow++;
                }
            }
            //------------------ Init complete
            //Start the training
            for (int epoch=0;epoch<_noOfEpoch; epoch++) //Do not parallelize
            {
                _learningRate = _alpha * (1-(epoch / _noOfEpoch));
                for (tdRow=0;tdRow<_noOfDataSamples;tdRow++) //For each value in trainingData row
                {
                    //Best matching Unit
                    //KeyValuePair<int, double>  bestMU = new KeyValuePair<int, double>(0,double.MaxValue);
                    KeyValuePair<int, double> [] allMU = 
                                    new KeyValuePair<int, double>[cbVector[0].Length];

                    //Should be in parallel
                    double[] dataTraining = GetLinearArray(_trainingData, tdRow, _trainingData.Length - 2);
                    //for (int row = 0; row < cbVector[0].Length; row++) //For each value in row
                    Parallel.For(0, cbVector[0].Length, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, row =>
                     {
                         double[] dataCBVector = GetLinearArray(cbVector, row, cbVector.Length - 2);
                         double distance = _distanceMeasure.getDistanceVector(dataTraining, dataCBVector);
                         allMU[row] = new KeyValuePair<int, double>(row, distance);
                     });

                    //Find the Best Matching Unit (BMU) which is at index 0 of list                    
                    List<KeyValuePair<int, double>> bmuList =
                                    GetSortedKeyValuePair(allMU);
                    int bmuRow = bmuList[0].Key;
                    //Now adjust the BMU
                    //Check class
                    //for (int col = 0; col < cbVector.Length-1; col++) //Dont change target value
                    Parallel.For(0, cbVector.Length - 1, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, col =>
                    {
                        //bmuj = bmuj - alpha  (trainingj - bmuj)
                        if (cbVector[_indexTargetAttribute][bmuRow]
                                == _trainingData[_indexTargetAttribute][tdRow]) // Same Class
                            cbVector[col][bmuRow] =
                                 cbVector[col][bmuRow] + _learningRate *
                                        (_trainingData[col][tdRow] - cbVector[col][bmuRow]);
                        else //Different class
                            cbVector[col][bmuRow] =
                                 cbVector[col][bmuRow] - _learningRate *
                                        (_trainingData[col][tdRow] - cbVector[col][bmuRow]);
                    }); //Adjust BMU
                } //For each value in trainingData row
            } //For each epoch
            
            //Now create the model
            ModelLVQ model = new ModelLVQ(_missingValue,
                                                indexTargetAttribute,
                                                cbVector.Length - 1,
                                                cbVector);
            return model;
        }


    }
}
