using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Concurrent;

namespace Dasmic.MLLib.Algorithms.DecisionTree
{
    public class FilteredData
    {
        private double [][] _filteredData;
        private ConcurrentBag<long> _trainingDataRowIndices;
        private long _numberOfRows;

        public FilteredData(double[][] filteredData,
                                ConcurrentBag<long> trainingDataRowIndices,
                                long numberOfRows)
        {
            _filteredData = filteredData;
            _trainingDataRowIndices = trainingDataRowIndices;
            _numberOfRows = numberOfRows;
        }

        public double[][] FilteredDataValues
        {
            get
            {
                return _filteredData;
            }
        }

        public ConcurrentBag<long> TrainingDataRowIndices
        {
            get
            {
                return _trainingDataRowIndices;
            }
        }

        public long NumberOfRows
        {
            get
            {
                return _numberOfRows;
            }
        }


    }
}
