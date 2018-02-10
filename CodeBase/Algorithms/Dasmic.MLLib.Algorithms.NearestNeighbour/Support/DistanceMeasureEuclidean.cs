using System;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Algorithms.NearestNeighbour
{
    public class DistanceMeasureEuclidean: IDistanceMeasure
    {
        private bool _useSqrt;

        /// <summary>
        /// Sets a flag to use Sqrt function for Euclidean distance
        /// 
        /// Not using Sqrt will improve performance
        /// </summary>
        /// <param name="UseSqrt"></param>
        public void setUseSqrt(bool UseSqrt)
        {
            _useSqrt = UseSqrt;
        }

        public double getDistanceVector(double [] a,
                                double [] b)
        {
            if (a.Length != b.Length)
                throw new InvalidDataException();
            double sum=0,tmpVal =0 ;

            for (int ii=0;ii<a.Length;ii++)
            {
                tmpVal = a[ii] - b[ii];
                sum += (tmpVal * tmpVal);
            }

            if (_useSqrt)
                return System.Math.Sqrt(sum);
            else
                return sum;
        }

    }
}
