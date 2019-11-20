import React, {Component} from "react";
import * as d3 from 'd3';
import * as topojson from 'topojson';

interface Props {
    width: number | string;
    height: number | string;
}

export default class Map extends Component<Props> {
    // @ts-ignore
    ref: SVGSVGElement;

    componentDidMount() {
        const projection = d3.geoMercator();
        const path = d3.geoPath().projection(projection);

        d3.json('./world-110m2.json').then(data => {

            console.log(data);
            d3.select(this.ref)
                .append("g")
                .selectAll("path")
                // @ts-ignore
                .data(topojson.feature(data, data.objects.countries).features)
                .enter().append("path")
                .attr("fill", "#69b3a2")
                // @ts-ignore
                .attr("d", path)
                .style("stroke", "#fff")
        });
    }

    render(): React.ReactElement {
        return (
            <svg className="container"
                 ref={(ref: SVGSVGElement) => this.ref = ref}
                 width={this.props.width}
                 height={this.props.height}>
            </svg>
        );
    }
}
