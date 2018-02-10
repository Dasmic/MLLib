using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.NearestNeighbour
{
    public interface IDistanceMeasure
    {
        //Return distance in single precision
        double getDistanceVector(double [] X, double [] Y);
    }
}
