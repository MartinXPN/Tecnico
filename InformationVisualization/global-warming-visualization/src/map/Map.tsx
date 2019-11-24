import React, {Component} from "react";
import * as d3 from 'd3';

interface Props {
    width: number | string;
    height: number | string;
    hoverCountry: (country: string | undefined) => void;
    currentHoveredCountry: string | undefined;
    selectedCountries: Set<string>;
    addCountry : (country: string) => void;
    removeCountry : (country: string) => void;
}

export default class Map extends Component<Props> {
    // @ts-ignore
    private ref: SVGSVGElement;
    // @ts-ignore
    private map: d3.Selection<SVGPathElement, unknown, SVGGElement, unknown>;


    componentDidUpdate(prevProps: Readonly<Props>, prevState: Readonly<{}>, snapshot?: any): void {
        const rect = this.ref.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;

        const projection = d3.geoMercator()
            .scale(150)
            .translate([w / 2, h / 1.5]);
        const path = d3.geoPath().projection(projection);

        const self = this;
        this.map
            .on('click', d => {
                // @ts-ignore
                const country = d.properties.name;
                if(this.props.selectedCountries.has(country))
                    this.props.removeCountry(country);
                else
                    this.props.addCountry(country);
            })
            .on('mouseover', d => {
                // @ts-ignore
                self.props.hoverCountry(d.properties.name);
            })
            .on('mouseout', d => {
                self.props.hoverCountry(undefined);
            })
            // @ts-ignore
            .attr('d', path)
            .style('fill', '#3B5988')
            .style('stroke', 'white')
            .style('opacity', (d: any) => d.properties.name === this.props.currentHoveredCountry ? 1 : 0.8)
            .style('stroke-width', (d: any) => d.properties.name === this.props.currentHoveredCountry ? 2 : 0.3);

    }

    componentDidMount() {
        d3.json('./world_countries.json').then(data => {
            this.map = d3.select(this.ref)
                .append("g")
                .attr('class', 'map')
                .selectAll("path")
                .data(data.features)
                .enter()
                .append("path");
        });
    }

    render(): React.ReactElement {
        return (
            <svg className="container"
                 ref={(ref: SVGSVGElement) => this.ref = ref}
                 width={this.props.width}
                 height={this.props.height}/>
        );
    }
}
