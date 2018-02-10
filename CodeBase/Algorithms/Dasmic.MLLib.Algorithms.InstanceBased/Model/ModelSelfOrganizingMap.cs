using System;
using System.Text;
using System.Threading.Tasks;
using Dasmic.MLLib.Common.Exceptions;
using Dasmic.Portable.Core;


namespace Dasmic.MLLib.Algorithms.InstanceBased
{
    public class ModelSelfOrganizingMap:ModelBase
    {
        public ModelSelfOrganizingMap(double missingValue,
                               int indexTargetAttribute, int countAttributes) :
                               base(missingValue, indexTargetAttribute, countAttributes)
        {

        }

        /// <summary>
        /// Indexed by xdim, ydim, weights
        /// </summary>
        public SingleSOMNode [][] SomMap;

        public override 
           double RunModelForSingleData(double[] data)
        {
            throw new Exception("Use RunModelForSingleData(double[] data, ref int X, ref int Y)");            
        }

        /// <summary>
        /// Returns the node which is closest to the data
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public 
          void RunModelForSingleData(double[] data, ref long X, 
                                                        ref long Y)
        {
            object mutex = new object();
            long tX=0, tY=0;
            Parallel.For(0, data.Length,
                       new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, xdim =>
                       //for (long xdim = 0; xdim < data.Length; xdim++) //Parallelize this loop
            {
                double computedDistValue = double.MaxValue;
                for (long ydim = 0; ydim < data.Length; ydim++)
                {
                    //Find best data row for this unit                   
                    computedDistValue =
                            SomMap[xdim][ydim].GetComputedDistance(data);
                    lock (mutex)
                    {
                        if (computedDistValue <= SomMap[xdim][ydim].NameIdDistance)
                        {
                            tX = xdim;
                            tY = ydim;
                        }
                    } //lock
                } //ydim                
            }); //xdim          
            X = tX;
            Y = tY;
        }


        /// <summary>
        /// Prints the SOM Map
        /// </summary>
        public StringBuilder GetPrintedSOMMap()
        {
            StringBuilder sb = new StringBuilder();
            string tmp;
            for (int xdim = 0; xdim < SomMap.Length; xdim++)
            { 
                for (int ydim = 0; ydim < SomMap[0].Length; ydim++)
                {
                    tmp = "";
                    for (int idx = 0; idx < SomMap[xdim][ydim].GetNoOfWeights(); idx++)
                    {
                        tmp = tmp + Math.Round(SomMap[xdim][ydim].GetWeight(idx),2).ToString() + " ";
                    }
                    //tmp = tmp + "\r\n";
                    sb.AppendLine(tmp);
                }
            }
            return sb;
        }

        /// <summary>
        /// Returns a SOM Map with Name Id's listed for each node
        /// rather than weights
        /// </summary>
        /// <param name="data">Data Matrix to compare weights against. Preffered value is the training data.Columns in data should be equal to number of weights </param>
        /// <param name="names">String name of each row in data</param>
        /// <returns>StringBuilder object with map</returns>
        public string GetPrintedSOMMapWithNameIds(double [][] data,
                                                            string [] names)
        {
            //Do checks
            if (data.Length != SomMap[0][0].GetNoOfWeights())
                throw new DataColumnMismatchException();
            if (data[0].Length != names.Length) //Name for each Row
                throw new DataColumnMismatchException();

            StringBuilder sb = new StringBuilder();

            Parallel.For(0, data.Length,
                       new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, xdim =>
            //for (long xdim = 0; xdim < SomMap.Length; xdim++) //Parallelize this loop
            {
                double computedDistValue = double.MaxValue;
                string tmpLine = "";
                for (long ydim = 0; ydim < SomMap[0].Length; ydim++)
                {
                    //Find best data row for this unit
                    for (int dataRow = 0; dataRow < data[0].Length; dataRow++)
                    {
                        double[] inputDataRow = SupportFunctions.GetLinearArray(data,
                                      dataRow, data.Length - 1);
                        computedDistValue =
                            SomMap[xdim][ydim].GetComputedDistance(inputDataRow);

                        if (computedDistValue <= SomMap[xdim][ydim].NameIdDistance)
                        {
                            SomMap[xdim][ydim].NameId = names[dataRow];
                            SomMap[xdim][ydim].NameIdDistance = computedDistValue;
                        }
                    }//DataRow
                    tmpLine += SomMap[xdim][ydim].NameId + " ";
                }//ydim
                sb.AppendLine(tmpLine);
            }); //xdim                        
            return sb.ToString();
        }
    }
}
