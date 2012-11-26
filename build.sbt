import AssemblyKeys._ // put this at the top of the file

name := "ontoner"

organization := "edu.umass.cs.iesl"

version := "0.1-SNAPSHOT"

scalaVersion := "2.9.1"

libraryDependencies += "org.scalatest" %% "scalatest" % "1.6.1" % "test"

libraryDependencies +=  "edu.umass.cs.iesl" %% "scalacommons" % "0.1-SNAPSHOT"  changing()

libraryDependencies +=  "cc.factorie" % "factorie" % "1.0-SNAPSHOT"  changing()

resolvers += "IESL Repo" at "https://dev-iesl.cs.umass.edu/nexus/content/repositories/releases"

resolvers += "IESL ThirdParty Repo" at "https://dev-iesl.cs.umass.edu/nexus/content/repositories/thirdparty"

resolvers += "IESL Snapshot Repo" at "https://dev-iesl.cs.umass.edu/nexus/content/repositories/snapshots"

resolvers += "David Soergel Repo" at "http://dev.davidsoergel.com/nexus/content/groups/public/"

resolvers += "David Soergel Snapshot Repo" at "http://dev.davidsoergel.com/nexus/content/repositories/snapshots"

resolvers += "Typesafe Repo" at "http://repo.typesafe.com/typesafe/releases/"



seq(assemblySettings: _*)

fork in run := true

fork in runMain := true

javaOptions in run += "-Xmx8G"

javaOptions in runMain += "-Xmx8G"


publishTo <<= (version)
                                            {version: String =>
                                              {
                                              def repo(name: String) = name at "https://dev-iesl.cs.umass.edu/nexus/content/repositories/" + name
                                              val isSnapshot = version.trim.endsWith("SNAPSHOT")
                                              val repoName = if (isSnapshot) "snapshots" else "releases"
                                              Some(repo(repoName))
                                              }
                                            }

credentials +=
                                  {
                                  Seq("build.publish.user", "build.publish.password").map(k => Option(System.getProperty(k))) match
                                  {
                                    case Seq(Some(user), Some(pass)) =>
                                      Credentials("Sonatype Nexus Repository Manager", "dev-iesl.cs.umass.edu", user, pass)
                                    case _ =>
                                      Credentials(Path.userHome / ".ivy2" / ".credentials")
                                  }
				  }
