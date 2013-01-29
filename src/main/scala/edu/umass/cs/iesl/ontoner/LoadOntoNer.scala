package edu.umass.cs.iesl.ontoner

/**
 * Created with IntelliJ IDEA.
 * User: sameyeam
 * Date: 1/29/13
 * Time: 2:39 AM
 */
import cc.factorie.CategoricalDomain
import cc.factorie.app.nlp.Token
import cc.factorie.app.nlp.ner.ChainNerLabel
import cc.factorie.app.nlp.Document
import cc.factorie.app.nlp.Sentence
import scala.collection.mutable.ArrayBuffer

object Conll2003NerDomain extends CategoricalDomain[String]
class Conll2003ChainNerLabel(token:Token, initialValue:String) extends ChainNerLabel(token, initialValue) {
  def domain = Conll2003NerDomain
}

object LoadOntoNer {
  import java.io.File
  def recursiveListFiles(f: File): Array[File] = {
    val these = f.listFiles
    these ++ these.filter(_.isDirectory).flatMap(recursiveListFiles)
  }

  def fromDirectory(directory:String) : Seq[Document] = {
    val files = recursiveListFiles(new File(directory)).filter(_.getAbsolutePath.endsWith(".pmd.n"))
    val docs = new ArrayBuffer[Document]
    files.foreach( x=> docs ++= fromFilename(x.getAbsolutePath) )
    docs.toSeq
  }

  def fromFilename(filename:String): Seq[Document] = {
    import scala.io.Source
    import scala.collection.mutable.ArrayBuffer

    val documents = new ArrayBuffer[Document]
    var document = new Document("CoNLL2003-"+documents.length, "")
    documents += document
    val source = Source.fromFile(new java.io.File(filename))
    var sentence = new Sentence(document)(null)
    for (line <- source.getLines()) {
      if (line.length < 2) { // Sentence boundary
        //sentence.stringLength = document.stringLength - sentence.stringStart
        //document += sentence
        document.appendString("\n")
        sentence = new Sentence(document)(null)
      } else if (line.startsWith("-DOCSTART-")) {
        // Skip document boundaries
        document = new Document("CoNLL2003-"+documents.length, "")
        documents += document
      } else {
        val fields = line.split('\t')
        val word = fields(0)
        val ner = fields(7).stripLineEnd
        if (sentence.length > 0) document.appendString(" ")
        val token = new Token(sentence, word)
        token.attr += new Conll2003ChainNerLabel(token, ner)
      }
    }
    println("Loaded "+documents.length+" documents with "+documents.map(_.sentences.size).sum+" sentences with "+documents.map(_.length).sum+" tokens total from file "+filename)
    documents
  }
}