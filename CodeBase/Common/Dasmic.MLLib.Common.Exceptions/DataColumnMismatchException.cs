using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Common.Exceptions
{
    public class DataColumnMismatchException : Exception
    {

        public DataColumnMismatchException(
            string message,
            Exception innerException) : base(message, innerException)
        {

        }


        public DataColumnMismatchException()
        {

        }

        public override string Message
        {
            get
            {
                return Resources.strings_messages.exception_data_column_mismatch;
            }
        }
    }
}
