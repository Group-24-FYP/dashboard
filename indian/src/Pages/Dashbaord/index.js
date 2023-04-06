import {
  DollarCircleOutlined,
  ShoppingCartOutlined,
  ShoppingOutlined,
  UserOutlined,
} from "@ant-design/icons";
import { Card, Space, Statistic, Table, Typography } from "antd";
import { useEffect, useState } from "react";
import { getCustomers, getInventory, getOrders, getRevenue } from "../../API";
import img1 from '../../image/bottle.png';
import video1 from '../../video/v1.mp4'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar } from "react-chartjs-2";
import ReviewsBar from "../../reviews/ReviewsBar";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

function Dashboard() {
  const [orders, setOrders] = useState(0);
  const [inventory, setInventory] = useState(0);
  const [customers, setCustomers] = useState(0);
  const [revenue, setRevenue] = useState(0);
  const [data , setData] = useState([{}])

  useEffect(() => {
    getOrders().then((res) => {
      setOrders(res.total);
      setRevenue(res.discountedTotal);
    });
    getInventory().then((res) => {
      setInventory(res.total);
    });
    getCustomers().then((res) => {
      setCustomers(res.total);
    });
  }, []);

  useEffect(() => {
    fetch("/members").then(
      (res) => res.json()
      ).then(
        data => {
          setData(data)
          console.log(data)
        }
      )
    
  }, []);

  return (
    <div>
      <h3></h3>
    <Space size={100} direction="horizontal" align="center">
      <div style={{ textAlign: 'center', 
              width: 850,
              height: 800,
              color: "red",
              backgroundColor: "rgba(255,0,0,0.25)",
              borderRadius: 20,
              fontSize: 24,
              padding: 20}}>
      <Typography.Title level={1}>Detection</Typography.Title>
      <Space>
        
      </Space>
      <Card style={{
              color: "red",
              backgroundColor: "rgba(255,255,0,0.25)",
              borderRadius: 50,
              fontSize: 24,
              padding: 0,
            }}>
        <h3>Status</h3>
        <h4>Defected Product</h4>
      </Card>

      <Card style={{
              width: '100%',
              height: '50%',
              color: "red",
              backgroundColor: "rgba(255,255,0,0.25)",
              borderRadius: 50,
              fontSize: 24,
              padding: 0,
            }}>
            <Space size={10} direction="vertical" align="baseline">
                <h2>Uncertainty Score</h2>
                <div  style={{
                      width: 200,
                      height: 200
                    }}>
                    
                      <ReviewsBar score='89.5' />
                </div>
              </Space>
      </Card>
      </div>
      
      <div style={{ textAlign: 'center', 
              width: 850,
              height: 800,
              color: "red",
              backgroundColor: "rgba(255,0,0,0.25)",
              borderRadius: 20,
              fontSize: 24,
              padding: 20,}}>
        <Typography.Title level={1}>Segmentation</Typography.Title>
        <div>
          <video width="400" height="200" controls>
                    <source src={video1} type="video/mp4" />
          </video>
        </div>
        <div>
          <h3>Anomaly Segmentation Map</h3>
          <img src={img1} alt=""/>
        </div>
      </div>
    </Space>
    </div>
    
    
  );
}

function DashboardCard({ title, value, icon ,img}) {
  return (
    <Card>
      <Space direction="horizontal">
        {icon}
        <Statistic title={title} value={value} />
      </Space>
      {img}
    </Card>
  );
}

function RecentOrders() {
  const [dataSource, setDataSource] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    getOrders().then((res) => {
      setDataSource(res.products.splice(0, 3));
      setLoading(false);
    });
  }, []);

  return (
    <>
      <Typography.Text>Recent Orders</Typography.Text>
      <Table
        columns={[
          {
            title: "Title",
            dataIndex: "title",
          },
          {
            title: "Quantity",
            dataIndex: "quantity",
          },
          {
            title: "Price",
            dataIndex: "discountedPrice",
          },
        ]}
        loading={loading}
        dataSource={dataSource}
        pagination={false}
      ></Table>
    </>
  );
}

function DashboardChart() {
  const [reveneuData, setReveneuData] = useState({
    labels: [],
    datasets: [],
  });

  useEffect(() => {
    getRevenue().then((res) => {
      const labels = res.carts.map((cart) => {
        return `User-${cart.userId}`;
      });
      const data = res.carts.map((cart) => {
        return cart.discountedTotal;
      });

      const dataSource = {
        labels,
        datasets: [
          {
            label: "Revenue",
            data: data,
            backgroundColor: "rgba(255, 0, 0, 1)",
          },
        ],
      };

      setReveneuData(dataSource);
    });
  }, []);

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: "bottom",
      },
      title: {
        display: true,
        text: "Order Revenue",
      },
    },
  };

  return (
    <Card style={{ width: 500, height: 250 }}>
      <Bar options={options} data={reveneuData} />
    </Card>
  );
}
export default Dashboard;
