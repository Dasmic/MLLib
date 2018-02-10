using System;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Algorithms.NearestNeighbour.Support
{
    public class DistanceMeasureManhattan : IDistanceMeasure
    {

        public double getDistanceVector(double[] a,
                                double[] b)
        {
            if (a.Length != b.Length)
                throw new InvalidDataException();
            double sum = 0, tmpVal = 0;
            for (int ii = 0; ii < a.Length; ii++)
            {
                tmpVal = a[ii] - b[ii];
                sum += System.Math.Abs(tmpVal);
            }

            return sum;
        }

    }
}
