using System;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Algorithms.InstanceBased
{
    public class SingleSOMNode
    {
        private double[] _weights;
        private long _X { get; set; }
        private long _Y { get; set; }

        //Identifying Name
        public string NameId { get; set; }
        public double NameIdDistance  { get; set; } //Store distance of current nameId

        public SingleSOMNode(long noOfWeights,long x,long y)
        {
            _weights = new double[noOfWeights];
            _X = x;
            _Y = y;
            NameIdDistance = double.MaxValue;
        }


        /// <summary>
        /// This function will be called from an external loop 
        /// to exploit parallelism
        /// </summary>
        /// <param name="noOfWeights"></param>
        /// <param name="rnd"></param>
        /// <param name="_weightBaseValue"></param>
        public void InitWeights(Random rnd,double _weightBaseValue)
        {
            for (int wIdx = 0; wIdx < _weights.Length; wIdx++)
            {
                _weights[wIdx] = 
                   rnd.Next(1, 99) * _weightBaseValue;  //Choose value between
            }
        }

        /// <summary>
        /// Computes Euclidean distance with weights
        /// 
        /// Data vector should be same length as weights
        /// </summary>
        /// <param name="inputData"></param>
        public double GetComputedDistance(double [] inputData)
        {
            if (inputData.Length != _weights.Length)
                throw new InvalidDataException();

            //Compute 
            double distance = 0;
            for (int wIdx = 0; wIdx < _weights.Length; wIdx++)
            {
                distance += Math.Pow((_weights[wIdx] - 
                                        inputData[wIdx]), 2.0);
            }
            return Math.Sqrt(distance);
        }

        public double GetWeight(int idxColumn)
        {
            return _weights[idxColumn];
        }

        public long GetNoOfWeights()
        {
            return _weights.Length;
        }

        /// <summary>
        /// Updates the node weights if it is within radius of BMU
        /// </summary>
        /// <param name="bmuNode"></param>
        /// <param name="inputData"></param>
        /// <param name="bmuRadius"></param>
        /// <param name="learningRate"></param>
        public void UpdateWeights(SingleSOMNode bmuNode, 
                    double [] inputData, 
                    double bmuRadius, 
                    double learningRate,
                    bool useDistanceInfluence=true)
        {
            if (inputData.Length != _weights.Length)
                throw new InvalidDataException();

            //if (Math.Round(_weights[0],2)==0)// && _X==0 && _Y==1)
            //    _X = 0;

            double distance = 0;
            distance = Math.Pow(_X - bmuNode._X, 2) + 
                                    Math.Pow(_Y - bmuNode._Y, 2);
            distance = Math.Sqrt(distance);

            double distInfluence;
            if (useDistanceInfluence)
            {
                distInfluence = Math.Pow(distance, 2);
                distInfluence = Math.Exp(-distInfluence /
                                      (2 * Math.Pow(bmuRadius, 2)));
            }
            else
            {
                distInfluence = 1.0;
            }
            
            //Weight should be increased proportional to distance from BMU
            if (distance <= bmuRadius) //Update Weights
            {
                for (int wIdx = 0; wIdx < _weights.Length; wIdx++)
                {
                    _weights[wIdx] += (inputData[wIdx] - _weights[wIdx]) 
                                        * learningRate * distInfluence;
                }
            }
        }
    }
}
