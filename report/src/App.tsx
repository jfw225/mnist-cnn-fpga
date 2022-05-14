import React, { useEffect, useState } from 'react';
import logo from './logo.svg';
import './App.css';
import Latex from "react-latex";
import ReactMarkdown from "react-markdown";

import ReportFile, { FetchReport } from "./ReportFile";
import MarkdownRender from './MarkdownRender';


function App() {
	// const [htmlFile, setHtmlFile] = useState({ __html: "<div> <h3> hi </h3> </div>" })
	const [htmlFile, setHtmlFile] = useState({ __html: ReportFile })
	const [MDFile, setMDFile] = useState("");

	useEffect(() => {
		FetchReport(setMDFile);
	}, [])
	console.log(MDFile);

	// return <MarkdownRender>{MDFile}</MarkdownRender>;
	// return <MarkdownRender children={"## Test\nHere is my name $x+5$"} />
	// return <MarkdownRender children={MDFile} />
	// return <MarkdownRender>{MDFile}</MarkdownRender>
	// return <ReactMarkdown>{MDFile}</ReactMarkdown>;
	// return <div dangerouslySetInnerHTML={htmlFile} ></div>
	return (
		<div className="App">
			<link
				href="//cdnjs.cloudflare.com/ajax/libs/KaTeX/0.9.0/katex.min.css"
				rel="stylesheet"
			/>
			<header className="App-header">
				<img src={logo} className="App-logo" alt="logo" />
				<MarkdownRender>{MDFile}</MarkdownRender>
			</header>
		</div>
	);
}

export default App;
