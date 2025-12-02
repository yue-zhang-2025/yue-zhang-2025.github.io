---
layout: home
title: Home
---

{% for post in site.posts %}
  <article>
    <h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
    <time>{{ post.date | date: "%B %d, %Y" }}</time>
    {{ post.excerpt }}
    <p><a href="{{ post.url }}">Read more →</a></p>
  </article>
  <hr>
{% endfor %}
