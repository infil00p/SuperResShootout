/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 ~ Copyright 2022 Adobe
 ~
 ~ Licensed under the Apache License, Version 2.0 (the "License");
 ~ you may not use this file except in compliance with the License.
 ~ You may obtain a copy of the License at
 ~
 ~     http://www.apache.org/licenses/LICENSE-2.0
 ~
 ~ Unless required by applicable law or agreed to in writing, software
 ~ distributed under the License is distributed on an "AS IS" BASIS,
 ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 ~ See the License for the specific language governing permissions and
 ~ limitations under the License.
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
package org.infil00p.superresstats

import android.util.JsonReader
import java.io.StringReader
import java.util.*

class ResultParser {

    fun loadResultsFromJson(results : String) : ResultSet
    {
        var reader = JsonReader(StringReader(results))
        var resultVector : Vector<Result>
        try {
            reader.beginObject()
            resultVector = readResults(reader)
        } finally {
            reader.close()
        }
        return ResultSet(resultVector)
    }

    private fun readResults(reader: JsonReader) : Vector<Result>
    {
        // Create empty vector
        var resultVector = Vector<Result>()
        // Consume the results object.
        if(reader.nextName().equals("results")) {
            reader.beginArray()
            while (reader.hasNext()) {
                lateinit var framework: String
                lateinit var filename: String
                lateinit var device: String
                var duration: Double
                duration = 0.0
                reader.beginObject()
                while (reader.hasNext()) {
                    var name = reader.nextName()
                    if (name.equals("framework")) {
                        framework = reader.nextString()
                    } else if (name.equals("duration")) {
                        duration = reader.nextDouble()
                    } else if (name.equals("imagePath")) {
                        filename = reader.nextString()
                    } else if (name.equals("device")) {
                        device = reader.nextString()
                    } else {
                        reader.skipValue()
                    }
                }
                reader.endObject()
                val result = Result(framework, duration, filename, device)
                resultVector.add(result)
            }
        }
        return resultVector
    }
}