package org.infil00p.superresstats

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.github.mikephil.charting.data.BarData
import com.github.mikephil.charting.data.BarDataSet
import com.github.mikephil.charting.data.BarEntry
import org.infil00p.superresstats.databinding.ActivityMainBinding
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // I'm assuming this does something
        var handler = AssetHandler(this);

        // Prep external storage permissions
        var external = this.getExternalFilesDir(null)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.button.setOnClickListener {
            if (external != null) {
                if (doTest(external.absolutePath)) {
                    // Get the file from internal storage
                    val tflite_results_file = File(external, "results_tflite.json")
                    val pytorch_results_file = File(external, "results_pytorch.json")
                    val ort_results_file = File(external, "results_ort.json")

                    val tfLiteText = tflite_results_file.readText();
                    val pytorchText = pytorch_results_file.readText();
                    val ortText = ort_results_file.readText();

                    // Parse these into JSON
                    val parser = ResultParser();
                    val tfLiteResults = parser.loadResultsFromJson(tfLiteText)
                    val pytorchResults = parser.loadResultsFromJson(pytorchText)
                    val ortResults = parser.loadResultsFromJson(ortText)

                    drawResults(tfLiteResults, pytorchResults, ortResults)

                }
            }
        }
    }

    fun drawResults(tfLiteResults: ResultSet, pyTorchResults: ResultSet, ortResults: ResultSet) {
        var barChart = binding.resultsBarChart
        val ourBarEntries: ArrayList<BarEntry> = ArrayList()

        val tfLiteMean = tfLiteResults.calculateAverage().toFloat()
        val pyTorchMean = pyTorchResults.calculateAverage().toFloat()
        val ortMean = ortResults.calculateAverage().toFloat()
        val tfliteBarEntry = BarEntry(0.0f, tfLiteMean)
        val pyTorchBarEntry = BarEntry(1.0f, pyTorchMean)
        val ortBarEntry = BarEntry(2.0f, ortMean)

        ourBarEntries.add(tfliteBarEntry)
        ourBarEntries.add(pyTorchBarEntry)
        ourBarEntries.add(ortBarEntry)

        val averageDataset = BarDataSet(ourBarEntries, "Average Inference Times")
        barChart.data = BarData(averageDataset)
        barChart.invalidate()
    }
    /**
     * A native method that is implemented by the 'superresshootout' native library,
     * which is packaged with this application.
     */
    external fun doTest(externalFilePath: String ): Boolean

    companion object {
        // Used to load the 'superresshootout' library on application startup.
        init {
            System.loadLibrary("superresstats")
        }
    }
}