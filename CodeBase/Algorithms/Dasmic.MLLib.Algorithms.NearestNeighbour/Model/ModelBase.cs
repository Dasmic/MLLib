using System.Collections.Generic;
using System.Threading.Tasks;
using Dasmic.MLLib.Common.MLCore;
using Dasmic.MLLib.Common.Exceptions;
using Dasmic.MLLib.Math.Statistics;
using Dasmic.Portable.Core;

namespace Dasmic.MLLib.Algorithms.NearestNeighbour
{  
    public class ModelBase: Common.MLCore.ModelBase
    {
        protected double[][] _data;
        protected int _k;
        protected IDistanceMeasure _distanceMeasure;

        public ModelBase(double missingValue,
                                int indexTargetAttribute,
                                int countAttributes,
                                double[][] data) :
                                base(missingValue,
                                    indexTargetAttribute,
                                    countAttributes)
        {
            DistanceMeasureEuclidean dme =
                new DistanceMeasureEuclidean();
            dme.setUseSqrt(false);
            _distanceMeasure = dme;
            
        }

        public override double RunModelForSingleData(double[] data)
        {
            double value = 0;
            VerifyDataForRun(data);
          
            if (data.Length != _data.Length - 1)
                throw new InvalidDataException();

            KeyValuePair<int, double>[] allValues =
                new KeyValuePair<int, double>[_data[0].Length];

            //Compute distance per Row
            Parallel.For(0, _data[0].Length, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, row =>
            {
                double[] tArray = SupportFunctions.GetLinearArray(_data, row, _data.Length - 2);               
                double distance = _distanceMeasure.getDistanceVector(tArray, data);
                allValues[row] = new KeyValuePair<int, double>(row, distance);
            });

            List<KeyValuePair<int, double>> distList =
                SupportFunctions.GetSortedKeyValuePair(allValues);

            double[] targetValues = new double[_k];
            //Get lowest k distancesPerRow and their count
            //for (int ii= 0; ii < _k;ii++)
            Parallel.For(0, _k, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, ii =>
            {
                int row = distList[ii].Key;
                targetValues[ii] = _data[_origTargetAttributeIndex][row];
            });

            //Get the mode
            value = Dispersion.Mode(targetValues);
            return value;
        }


        //Serialization Routine
        public override void SaveModel(string filePath)
        {

        }

        //Deserialization Routine
        public override void LoadModel(string filePath)
        {

        }




    }
}
