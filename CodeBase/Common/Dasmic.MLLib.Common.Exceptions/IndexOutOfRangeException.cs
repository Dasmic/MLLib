using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Common.Exceptions
{
    public class IndexOutOfRangeException : Exception
    {
        public IndexOutOfRangeException(
            string message,
            Exception innerException) : base(message, innerException)
        {

        }


        public IndexOutOfRangeException()
        {

        }

        public override string Message
        {
            get
            {
                return Resources.strings_messages.exception_invalid_data;
            }
        }
    }
}
