import React, { useEffect, useState } from 'react';
import logo from './logo.svg';
import './App.css';
import Latex from "react-latex";

import ReportFile from "./ReportFile";

// import report from "./src/report-jup.html";
async function loadReportFile(setHtmlFile: any) {
	// fetch the report file
	const file = await fetch("./public/report-jup.html");
	console.log(file)

	// convert the report file to a string
	const file_str = await file.text();

	console.log(file_str);
	// set the report element
	setHtmlFile({ __html: file_str });
}

function App() {
	// const [htmlFile, setHtmlFile] = useState({ __html: "<div> <h3> hi </h3> </div>" })
	const [htmlFile, setHtmlFile] = useState({ __html: ReportFile })

	useEffect(() => {
		// loadReportFile(setHtmlFile);
	}, [])

	console.log(htmlFile)

	return (
		<div className="App">


			<link
				href="//cdnjs.cloudflare.com/ajax/libs/KaTeX/0.9.0/katex.min.css"
				rel="stylesheet"
			/>
			<header className="App-header">
				<img src={logo} className="App-logo" alt="logo" />
				<p>
					Edit <code>src/App.tsx</code> and save to reload.
				</p>
				<div dangerouslySetInnerHTML={htmlFile} ></div>
				<h2>
					<Latex> What is $3\times3$?</Latex>
				</h2>
				<a
					className="App-link"
					href="https://reactjs.org"
					target="_blank"
					rel="noopener noreferrer"
				>
					Learn React
				</a>
			</header>
		</div>
	);
}

export default App;
