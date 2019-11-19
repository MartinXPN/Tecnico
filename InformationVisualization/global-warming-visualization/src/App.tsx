import React, {Component} from 'react';
import logo from './logo.svg';
import './App.css';
import * as d3 from 'd3'

interface Props {
}

interface State {
    sea2glaciers: d3.DSVParsedArray<{ year: Date, level: number, mass: number }>;
    data: d3.DSVParsedArray<{ country: string, year: Date, gdp: number, meat_consumption: number, temperature: number, ghg_emission: number }>
    yearStart: number;
    yearEnd: number;
    selectedCountries: Array<string>;
}

export default class App extends Component<Props, State> {

    componentDidMount(): void {
        d3.json('./sea2glaciers.json').then(data => {
            this.setState({sea2glaciers: data});
            console.log(data);
        });

        d3.json('./gdp2temp2meat2ghg.json').then(data => {
            this.setState({data: data});
            console.log(data);
        });
    }

    render(): React.ReactElement {
        return (
            <div className="App">
                <header className="App-header">
                    <img src={logo} className="App-logo" alt="logo"/>
                    <p>
                        Edit <code>src/App.tsx</code> and save to reload.
                    </p>
                    <a
                        className="App-link"
                        href="https://reactjs.org"
                        target="_blank"
                        rel="noopener noreferrer">
                        Learn React
                    </a>
                </header>
            </div>
        );
    }
};
